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


train_path = '../PART_1/Data/dogs-vs-cats/train'
valid_path = '../PART_1/Data/dogs-vs-cats/valid'
test_path = '../PART_1/Data/dogs-vs-cats/test'

target_size = (224, 224) # 224 by 224 px
prep = keras.applications.mobilenet.preprocess_input

train_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory=train_path,
                        target_size=(224,224), classes=['cat','dog'], batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory=valid_path,
                        target_size=target_size, classes=['cat','dog'], batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function = prep).\
    flow_from_directory(directory= test_path,
                        target_size=target_size, classes=['cat','dog'], batch_size=10, shuffle=False)

mobile = keras.applications.mobilenet.MobileNet()
mobile.summary()

## Modify Model
x = mobile.layers[-6].output
predictions = Dense(2, activation="softmax")(x)
#"Model" from keras functional[!Sequential] API
model = Model(inputs = mobile.input, outputs=predictions)

# Fine-Tuning our Output layer
for layer in model.layers[:-5]:
    layer.trainable = False

## Training
### COMPILE
opt = Adam(lr= 0.0001)
loss = 'categorical_crossentropy'
metrics =['accuracy']
model.compile(optimizer = opt, loss = loss, metrics = metrics)

model.fit_generator(train_batches, steps_per_epoch=10,
                        validation_data=valid_batches, validation_steps=5, epochs=30, verbose=2)

test_labels = test_batches.classes
test_labels
test_batches.class_indices

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
plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm = cm,
                      classes = plot_labels, title='Confusion Matrix')
plt.show()
