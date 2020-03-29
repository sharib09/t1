import tensorflow as tf
from tensorflow.keras.models import load_model

import pickle
import cv2
flatten= -1



# load the input image and resize 
path = r'home/pi/tensorflow_test/test_image/apple.png'
image = cv2.imread(path)
output = image.copy()
image = cv2.resize(image, (32, 32))
flatten=1

# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0

# check to see if we should flatten the image and add a batch
# dimension
if flatten > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))

# we will flatten the image as we are not using convolutional neural network
else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model=tf.keras.models.load_model(r'home/pi/tensorflow_test/neuralNetModel')

lb = pickle.loads(open("neuralNetlb.pickle", "rb").read())

# make a prediction on the image
preds = model.predict(image)

# find the class label index with the largest corresponding probabbility

i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 0, 255), 2)

# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)
