from keras import layers
from keras import models
import keras
from keras import callbacks
from keras import initializers
from keras.layers import AveragePooling2D, Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU as LR
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import plot_model
from matplotlib import pyplot
import matplotlib.patches as mpatches
import numpy as np
from scipy.misc import imsave
from numpy.random import seed
from time import time



seed(1)   	#Neural network algorithms are stochastic. This means they make use of randomness, such as initializing to random weights, and in turn the same network trained on the same data can produce different results. You can seed the random number generator so that you can get the same results from the same network on the same data, every time. The seed() method is used to initialize the random number generator, which needs a number to start with (a seed value). This specified seed value, such as “1”, ensures that the same sequence of random numbers is generated each time the code is run.
batch_size = 4   #The training is performed in batches of four images. This means that four images are forward fed into the network and then the gradient is computed. Training a network with batches helps in reducing training time to the detriment of the accuracy. In this specific case we have about 1000 images, which is not a huge number to train this network. So, mini-batches of four images is an optimal value to accelerate the training. If you have a different number of training images try to change the size of the mini-batch.
img_height = 650  #A CNN requires a fixed input image size. This is the image size that gave us the best accuracy results with our training dataset. If you have your own dataset you can try other image sizes to study how the accuracy varies with the image dimension. You can increase the image size and improve the image resolution, but paying attention to the limits of the GPUs RAM.
img_width = 650
img_channels = 3  #This is number of channels of the images. Mammographic exams are grayscale images and, therefore, they have only one channel. However, Resnet models are fine-tuned for using 3 channels, because they are designed to work on RGB images. When you set the number of channels to 3, the conversion from 1 to 3 channels is done automatically by the ImageDataGenerator class provided by Keras at training time, by identically repeating the same pixel value in all 3 channels. In other words, the image is a tensor composed of 3 identical 2D matrices.

#
# network params
#

cardinality = 1  #Cardinality of a ResNeXt. For ResNet set `cardinality` = 1.


def residual_network(x):

    """
     Defines the architecture of the residual convolutional neural network (ResNet).

     Parameters
     ----------
     x : input of the network, tf.Tensor
	The input image of shape=(img_height, img_width, img_channels).

     Returns
     -------
     x : output of the network, tf.Tensor
        A tensor (1,4). The type of each element of the tensor is a float32 and it is the probability of the correspondent class.
    
    """

    def add_common_layers(y):

        """
	 Defines a couple of subsequent layers frequently used through the model.
	 Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1. The input is a tensor, the output is a tensor of same shape as 		 input.
	 LeakyReLU is the activation function. The default slope coefficient is set to 0.3.

	 Parameters
         ----------
         y : input, tf.Tensor

	 Returns
     	 -------
     	 y : output of the LeakyReLU layer, tf.Tensor
            Same shape as the input.
	    	
	"""
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        return y

    def grouped_convolution(y, nb_channels, _strides):

	"""
	 When `cardinality` == 1 this is just a standard convolution.
	 This function is called in "residual_block" function.
	    	
	"""
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.

	The input of each of the residual block is shared by two branches: in the first, it passes through several convolutional, batch normalization, activation and max pooling layers while in the other 		branch it passes through a convolutional layer and a batch normalization only. The outputs of these two branches are then added together to constitute the residual block and then LeakyReLu is 	performed on the the output of the residual block.

        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by  convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # Now we can use the defined functions "add_common_layers" and "residual_block" to build the architecture of our ResNet model.

    # conv1
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)   #The first 2D Convolution has 64 filters of shape (3,3) and uses a stride of (2,2). Its name is "conv1".
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)   #MaxPooling on the output of conv1. It uses a (3,3) window and a (2,2) stride. The resulting output shape when using the 											"same" padding option is: output_shape = input_shape / strides.
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 64, 128, _project_shortcut=project_shortcut)    #1st residual block

    # conv3
    for i in range(3):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 256, _strides=strides)      #2nd residual block

    # conv4
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 256, _strides=strides)      #3rd residual block

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 256, _strides=strides)      #4th residual block
    
    x = Dropout(0.2)(x)    #Dropout layer with rate (fraction of the input units to drop) set to 0.2. You can try with a different dropout rate.

    x = layers.GlobalAveragePooling2D()(x)    #Global Average Pooling layer that calculates the average output of each feature map in the previous layer. In this case the output size is (1, 256).
  
    x = layers.Dense(4, activation = 'softmax')(x)  #Dense layer with 4 units and with Softmax as activation function. The input is the output tensor of the GAP layer. The output is a tensor of size (1, 							     4) whose elements are float32 and they are the probabilities of each of the four classes.

    return x


image_tensor = layers.Input(shape=(img_height, img_width, img_channels))   #We instantiate a Keras tensor as input of the network.
network_output = residual_network(image_tensor)    
  
model = models.Model(inputs=[image_tensor], outputs=[network_output])   #Once instantiated the model architecture and the input, we can create the model with this class that groups layers into an object 										 with training and inference features.
print(model.summary())    #We print a summary of the created model, with the output size and the number of parameters for each layer.
    

# definition of lr of the optimizer
rmsprop = optimizers.RMSprop(lr=0.1)    #Optimizer that implements the RMSprop algorithm, with a learning rate of 0.1.

# definition of callbacks

reduce_lr_val = ReduceLROnPlateau(monitor='val_loss', factor=0.1,              
                   patience=15, min_lr=0.001, verbose = 1)          ##This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced by a 										certain factor until a minimum value.
           

filepath="CC_R_model/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"         #Path to save the model file
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')                          #Callback to save the Keras model or model weights at some frequency.
callbacks_list = [checkpoint]

def on_epoch_end(self, epoch, logs=None):
    print(keras.eval(self.model.optimizer.lr))              #This is called at the end of an epoch during training.



sgd =keras.optimizers.SGD(lr=0.1, decay=1e-1, momentum=0.9, nesterov=True)     #Gradient descent (with momentum) optimizer.

# definition of the inputs via ImageDataGenerator

train_datagen = ImageDataGenerator(              #The built-in class ImageDataGenerator generates batches of tensor image data with real-time data augmentation each time the data are looped over, which 							   means that images are randomly augmented at runtime. This class also allows defining the image normalization (rescale).
	rescale=1./255,
        zoom_range=0.2,
	samplewise_center = True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=10,
        horizontal_flip=False)                   #We set the horizontal flip option to false because we train the model on a single projection at a time. So, by flipping the image horizontally we obtain 							  another view. You can try a different data augmentation by changing the other class arguments.


val_datagen = ImageDataGenerator(
        samplewise_center = True,
	rescale=1./255)
   

train_generator = train_datagen.flow_from_directory(                                            #Takes the path to a directory and generates batches of augmented data.
        '/TrainingSet/CC_R/train',   #Insert your own path to the directory containing the training set of images of the selected projection. This 													model is trained on a single projection at a time. In this example we train the model on the CC_R 													projection. Then train the model separately on the other projections (CC_L, MLO_R, MLO_L). 										
        batch_size = batch_size,
        target_size=(img_width, img_height),
        color_mode = 'rgb',     #If you set the number of channels to 3, you have to set color_mode ='rgb', when you use 1 single channel, you have to set color_mode = 'grayscale'.
        shuffle = True,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        '/ValidationSet/CC_R/validation', #Insert your own path to the directory containing the validation set of images of the selected projection (e.g.CC_R)
        batch_size = batch_size,
        target_size=(img_width, img_height),
        color_mode = 'rgb',
        shuffle = True,
        class_mode='categorical')

tensorboard = TensorBoard(log_dir="CC_R_model/logs/{}".format(time()), histogram_freq=0, batch_size=4, write_grads=True, write_images=True)   #This is 															optional. It enables visualizations for TensorBoard, a visualization tool provided with 															TensorFlow. It could be useful to visualize the model graph to check that the 																trained model’s structure matches our intended design, if the layers are built 																correctly and the shapes of inputs and outputs of the nodes are those expected. 															Insert tensorboard as callback in model.fit_generator. After training, if you have 																installed TensorFlow with pip, you should be able to launch TensorBoard from the 																command line: tensorboard --logdir=path_to_your_logs

model.compile(optimizer='sgd',                    #Configures the model for training.
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,    #Fits the model on data yielded batch-by-batch by a Python generator.
          steps_per_epoch=227,           #It is the ratio between the number of images in the training set and the batch size. Approximated to the nearest integer by default.
          epochs = 100,
          validation_data=validation_generator,
          validation_steps=38,              #It is the ratio between the number of images in the validation set and the batch size. Approximated to the nearest integer by default.
          callbacks=[reduce_lr_val, checkpoint, tensorboard])


np.savetxt('CC_R_model/loss_train.txt', history.history['loss'], delimiter=",")
np.savetxt('CC_R_model/acc_train.txt', history.history['acc'], delimiter=",")
np.savetxt('CC_R_model/loss_val.txt', history.history['val_loss'], delimiter=",")
np.savetxt('CC_R_model/acc_val.txt', history.history['val_acc'], delimiter=",")


model.save('CC_R_model/final_weights.h5')

