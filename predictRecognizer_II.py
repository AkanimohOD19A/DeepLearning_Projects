import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

classifier = load_model('VGG_2Block.h5')

## III: Making new Predictions

test_image = load_img("C:/Users/HP/Pictures/SGOD/18.jpg",
                            target_size=(64, 64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
print(round(result[0][0]))
#training_set.class_indices
if result[0][0] == 1:
    print('prediction = dog')
else:
    print('prediction = cat')

## fit model
	#history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
	#	validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
	#_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	#print('> %.3f' % (acc * 100.0))
	# learning curves
	#summarize_diagnostics(history)