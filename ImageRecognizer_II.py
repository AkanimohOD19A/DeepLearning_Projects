##Importing Keras Libraries and Packages

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

## Initialising the Convolutional Neural Network
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation = 'relu')) #1. Convolution
classifier.add(MaxPooling2D(pool_size = (2, 2))) # Pooling

# #2. Adds a Second Convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten()) #3. Flattens the Dataset

# #4. Full Connection
classifier.add(Dense(units = 128, activation = 'relu')) #units, the number of nodes in the layer
classifier.add(Dense(units = 1, activation = 'sigmoid')) #output, should have just one unit, think the 'sigmoid' for binary classification

# #5. Compiling the CNN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy', metrics  = ['accuracy'])

## II: Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# #1. Pre-Processing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)
### Aside the 'rescale', the other arguments are performed for data augmentation
# -blurring, zooming and flipping respectively;
# width_shift_range=0.1, height_shift_range=0.1.

training_set = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train', target_size=(64, 64),
                                                 batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test', target_size= (64, 64),
                                            batch_size=32, class_mode='binary')
### 'target_size' load images with 64 by 64 pixels

# #/FITTING
classifier.fit_generator(training_set, steps_per_epoch=8000,
                         epochs = 25, validation_data=test_set,
                         validation_steps=2000) #? validation_steps

## Save the model
classifier.save('C:/Users/HP/PycharmProjects/Machine Learning/Image_Recognition/VGG_2Block.h5')
