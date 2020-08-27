#Dogs vs Cats

import matplotlib.pyplot as plt
from matplotlib.image import imread

### data location
folder = 'C:/Users/HP/Downloads/dogs-vs-cats/train/train/'

### plot first few images
for i in range(9):
    #define subplt
    plt.subplot(330 + 1 + i)
    ##define filename
    filename = folder + 'cat.' + str(i) + '.jpg'
    ##load image pixels
    image = imread(filename)
    ##plot raw px data
    plt.imshow(image)
# show the figure
plt.show()

## Preprocessing into Standard Directories
from os import makedirs, listdir
from shutil import copyfile
from random import seed
from random import random

### create dir
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labeldir in labeldirs:
        newdir = dataset_home + subdir + labeldir
        makedirs(newdir, exist_ok=True)

### set random number generator
seed(1)
##3 define ratio of pictures to use for validation
val_ratio = 0.25
### copy training dataset images into sub-directories
src_directory = 'C:/Users/HP/Downloads/dogs-vs-cats/train/train/'

## Develop a Baseline CNN Model
import sys
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer= 'he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum = 0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

## plot diagnostic learning curve
def summarize_diagnostic(history):
    #plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color = 'blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label = 'test')
    #plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color = 'orange', label = 'test')
    #save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()

## evaluation
def run_test_harness():
    #define the model
    model = define_model()
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                           class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                          class_mode='binary', batch_size=64, target_size=(200, 200))
    # fit model
    history = model.fit_generator(train_it, steps_per_epoch = len(train_it),validation_data=test_it,
                                  validation_steps=len(test_it), epochs=20, verbose=0)

    ## Evaluating with our validation Data
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    ## Learning curves
    summarize_diagnostic(history)

## Run
run_test_harness()
