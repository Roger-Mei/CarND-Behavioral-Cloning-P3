import csv
from scipy import ndimage
import numpy as np
import cv2


################################################################################
# Load the data
################################################################################

lines = []
with open('../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
correction = 0.4
for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../../../opt/carnd_p3/data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        if i == 0:
            measurement = float(line[3])
            measurements.append(measurement)
        if i == 1:
            measurement = float(line[3]) - correction
            measurements.append(measurement)
        if i == 2:
            measurement = float(line[3]) + correction
            measurements.append(measurement)

################################################################################
# Data Augmentation
################################################################################
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
################################################################################
# Network Architecture (LeNet)
################################################################################

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5)) # Normalizing the data
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

model.save('model.h5')
