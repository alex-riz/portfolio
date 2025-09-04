from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

traindata = ImageDataGenerator()

data = traindata.flow_from_directory('GameImages/training', target_size=(800,800),batch_size = 32, class_mode='binary')
data.class_indices = {'peppa': 0, 'assassin':1}

validation = traindata.flow_from_directory('GameImages/tests', target_size=(800,800),batch_size = 32, class_mode='binary')
validation.class_indices = {'peppa': 0, 'assassin':1}

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3), activation='relu',input_shape=(800, 800,3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout (0.25))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout (0.25))
model.add(Flatten())
model.add(Dense (64, activation='relu'))
model.add(Dropout (0.5))
model.add(Dense (1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(data, steps_per_epoch=8, epochs = 10,validation_data = validation, validation_steps=2, batch_size=32)

model.save('my_model.h5')