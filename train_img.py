
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths

import numpy as np
import random
import pickle
import cv2
import os
print("[INFO] loading images...")
data = []
labels = []
imageLoc = sorted(list(paths.list_images(r"/home/pi/tensorflow_test/train_data")))
random.seed(42)
random.shuffle(imageLoc)
for imagePath in imageLoc:
	# load the image, and resize to 32x32 pixels 
	#  flatten the image into 32x32x3=3072 pixel 
	
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)
    
    # create labels 
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.25, random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
model = tf.keras.Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))



# initialize our initial learning rate 
INIT_LR = 0.01
EPOCHS = 75
# compile the model using SGD 

print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

N = np.arange(0, EPOCHS)

print("[INFO] serializing network and label binarizer...")
model.save(r"/home/pi/tensorflow_test/neuralNetModel")
f = open("neuralNetlb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()
