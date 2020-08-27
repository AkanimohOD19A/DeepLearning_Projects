## Making Prediction
### Loading librabries
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

### Load and Prepare the Image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

### load an image and predict the class
def pred_img(image_path):
    img = load_image(image_path)
    model = load_model('C:/Users/HP/PycharmProjects/Machine Learning/TensorFlow/L_CNN_model.h5')
    digit = model.predict_classes(img)
    print(digit)

pred_img('C:/Users/HP/Desktop/dad.png')

