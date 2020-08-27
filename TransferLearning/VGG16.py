import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



train_path = '../PART_1/Data/dogs-vs-cats/train'
valid_path = '../PART_1/Data/dogs-vs-cats/valid'
test_path = '../PART_1/Data/dogs-vs-cats/test'

target_size = (224, 224) # 224 by 224 px
prep = tf.keras.applications.vgg16.preprocess_input

train_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory=train_path,
                        target_size=(224,224), classes=['cat','dog'], batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory=valid_path,
                        target_size=target_size, classes=['cat','dog'], batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory= test_path,
                        target_size=target_size, classes=['cat','dog'], batch_size=10, shuffle=False)

#Verifying our Batches
assert train_batches.n == 1000
assert valid_batches.n == 100
assert test_batches.n == 200
assert train_batches.num_classes == test_batches.num_classes == valid_batches.num_classes == 2

##
imgs, labels = next(train_batches)

##Plot images as grid with 1 row and 10 cols
def plotImages(images_Arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_Arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# TRANSFER LEARNING FROM VGG16
vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()

### Corce VGG to Sequential and remove the output layer
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

###Disable ReTraining the layers
for layer in model.layers:
    layer.trainable = False

###Add our own Output layer, Parameters
model.add(Dense(units=2, activation="softmax"))
model.summary()

## Compile
opt = Adam(learning_rate= 0.0001)
loss = 'categorical_crossentropy'
metrics =['accuracy']
model.compile(optimizer = opt, loss = loss, metrics = metrics)

## Saving
from tensorflow.keras.models import load_model
import os.path
if os.path.isfile('../PART_1/Models/FineTuneVGG16_model.h5') is False:
    model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)
    model.save('../PART_1/Models/FineTuneVGG16_model.h5')
else:
    model = load_model('../PART_1/Models/FineTuneVGG16_model.h5')

## Making Inference
test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
test_labels

test_labels = test_labels[:,0]
print(test_labels)

predictions = model.predict(x = test_batches, steps=1, verbose =0)
np.round(predictions)

#Visualizing Confusion Matrix
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_true=test_labels,
                      y_pred=np.round(predictions[:,0]))
plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm = cm,
                      classes = plot_labels, title='Confusion Matrix')
plt.show()
#Perfomed wonderfully @ 100%, the concern is the limitation of sample size
if os.path.isfile('../PART_1/Graphs/FineTuneVGG16_graph.png') is False:
    plt.savefig('../PART_1/Graphs/FineTuneVGG16_graph.png', transparent=True)