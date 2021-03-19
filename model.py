import csv
from scipy import ndimage
import numpy as np
import cv2
import sklearn
from sklearn.utils import shuffle

################################################################################
# Utility functions
################################################################################

def generate_training_data(samples, batch_size=128):
    '''
    This function create the generator to deal with the large scale data
    '''
    num_smaples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_smaples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            measurements = []

            correction = 1.2
            for line in samples:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = '../../../opt/carnd_p3/data/IMG/' + filename
                    image = ndimage.imread(current_path)
                    images.append(image)
                    images.append(cv2.flip(image, 1)) # Data augmentation by flipping the image
                    # Central image
                    if i == 0: 
                        measurement = float(line[3])
                        measurements.append(measurement)
                        measurements.append(measurement*-1.0) # Data augmentation by flipping the image
                    # Left image
                    if i == 1:
                        measurement = float(line[3]) + correction
                        measurements.append(measurement)
                        measurements.append(measurement*-1.0) # Data augmentation by flipping the image       
                    # Right image
                    if i == 2:
                        measurement = float(line[3]) - correction
                        measurements.append(measurement) 
                        measurements.append(measurement*-1.0) # Data augmentation by flipping the image         

        X_train = np.array(images)
        y_train = np.array(measurements)

        yield sklearn.utils.shuffle(X_train, y_train)
################################################################################
# Load the data
################################################################################

lines = []
with open('../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

batch_size = 100  

train_generator = generate_training_data(train_samples, batch_size=batch_size)
validation_generator = generate_training_data(validation_samples, batch_size=batch_size)


# lines = []
# with open('../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)

# images = []
# measurements = []
# correction = 0.25
# for line in lines[1:]:
#     for i in range(3):
#         source_path = line[i]
#         filename = source_path.split('/')[-1]
#         current_path = '../../../opt/carnd_p3/data/IMG/' + filename
#         image = ndimage.imread(current_path)
#         images.append(image)
        
#         if i == 0:
#             measurement = float(line[3])
#             measurements.append(measurement)
#             measurements.append(measurement*-1.0)
#         if i == 1:
#             measurement = float(line[3]) - correction
#             measurements.append(measurement)
#             measurements.append(measurement*-1.0)
#         if i == 2:
#             measurement = float(line[3]) + correction
#             measurements.append(measurement)
#             measurements.append(measurement*-1.0)

# ################################################################################
# # Data Augmentation
# ################################################################################
# augmented_images, augmented_measurements = [], []
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image, 1))
#     augmented_measurements.append(measurement*-1.0)


# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)
################################################################################
# Network Architecture 
################################################################################

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

################################### LeNet ######################################
model = Sequential()
model.add(Cropping2D(cropping = ((50, 20), (0, 0)), input_shape=(160,320,3))) # Cropping the data to reduce the distraction
model.add(Lambda(lambda x: x/255.0 - 0.5)) # Normalizing the data
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# ################################### NVDIA CNN ######################################
# model =Sequential()
# model.add(Cropping2D(cropping = ((50, 20), (0, 0)))) # Cropping the data to reduce the distraction
# model.add(Lambda(lambda x: x/255.0 - 0.5)) # Normalizing the data
# model.add(Convolution2D(24, 5, 2, activation = "relu"))
# model.add(Convolution2D(36, 5, 2, activation = "relu"))
# model.add(Convolution2D(48, 5, 2, activation = "relu"))
# model.add(Convolution2D(64, 3, 1, activation = "relu"))
# model.add(Convolution2D(64, 3, 1, activation = "relu"))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))



# model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 4)
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, \
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=4, verbose=1)

model.save('model.h5')
