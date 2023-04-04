import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model("keras_model.h5")

camera = cv2.VideoCapture(0)

while True:

	status , frame = camera.read()

	if status:
                
		frame = cv2.flip(frame , 1)
                
		img = cv2.resize(frame,(224,224))
		
		testImage = np.expand_dims(img, axis = 0)
                
		normalisedImage = testImage/255.0
	
		prediction = model.predict(normalisedImage)
		
		cv2.imshow('feed' , frame)

		code = cv2.waitKey(1)
		
		if code == 32:
			break

camera.release()

cv2.destroyAllWindows()





