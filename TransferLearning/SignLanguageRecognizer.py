# Organize data into train, valid, test dirs
import os, shutil
import random
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


train_path = '../PART_1/Data/Sign-Language-Digits-Dataset-master/Dataset/train'
valid_path = '../PART_1/Data/Sign-Language-Digits-Dataset-master/Dataset/valid'
test_path  = '../PART_1/Data/Sign-Language-Digits-Dataset-master/Dataset/test'

target_size = (224, 224) # 224 by 224 px
prep = keras.applications.mobilenet.preprocess_input

train_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory=train_path,
                        target_size=(224,224), batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory=valid_path,
                        target_size=target_size, batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory= test_path,
                        target_size=target_size, batch_size=10, shuffle=False)

mobile = keras.applications.mobilenet.MobileNet()

x = mobile.layers[-6].output
predictions = Dense(10, activation="softmax")(x)
model = Model(inputs = mobile.input, outputs=predictions)

for layer in model.layers[:-23]:
    layer.trainable = False

opt = Adam(lr= 0.0001)
loss = 'categorical_crossentropy'
metrics =['accuracy']
model.compile(optimizer = opt, loss = loss, metrics = metrics)

from keras.models import load_model
import os.path
path = '../PART_1/Models/FineTuneMobileNet_SignLanguage_model.h5'
if os.path.isfile(path) is False:
    model.fit_generator(train_batches, steps_per_epoch=len(train_batches),
                        validation_data=valid_batches, validation_steps=len(valid_batches),
                        epochs=60, verbose=2)
    model.save(path)
else:
    model = load_model(path)


test_labels = test_batches.classes

## PREDICT
predictions = model.predict_generator(test_batches, steps = len(test_batches))

## CONFUSION MATRIX
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

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_confusion_matrix(cm = cm,
                      classes = plot_labels, title='Confusion Matrix')
plt.show()

