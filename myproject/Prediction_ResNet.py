from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# dimensions of our images

img_width, img_height = 650 , 650    #Set the image size you used to train the network.

test_datagen = ImageDataGenerator(
	samplewise_center = True,
        rescale=1./255)                  #This is the normalization on the images to use in the test set. Use the same normalization used on the training images.
        
test_generator = test_datagen.flow_from_directory(
        'TestSet/CC_R/test',  # this is the target directory, where the test set of images is. Select the same projection on which you trained the model you want to test.
        target_size=(img_width, img_height),  
        batch_size=1,
        color_mode = 'rgb',      #Use 'rgb' as color_mode if you trained the network on images with 3 channels. If you trained the network on images with 1 channel, use color_mode='grayscale'.
        shuffle = False,
        class_mode='categorical') 
        


# load the model we saved

model = load_model('CC_R_model/weights-improvement-46-0.80.h5')   #Insert the path to the model correspondent to the epoch that produced the best validation 															   accuracy improvement.

model.compile(loss='categorical_crossentropy',     #Configures the model for testing.
              optimizer='rmsprop',
              metrics=['accuracy'])
classes = model.predict_generator(test_generator, 128)        #Generates predictions for the input samples from a data generator. 128 is the number of test images. Change it with the number of your images.
print(classes, classes.shape, np.argmax(classes, axis = 1))       #Classes is a tensor of size (number of test images x 4), in each row there is the probability of each of the 4 classes for that image.
np.savetxt('/Predictions/predictions_ccr.txt', classes)    #Save the results of prediction as a .txt file.
