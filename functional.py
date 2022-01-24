#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

IMG_SIZE = 150

resize_and_rescale = Sequential([
    layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    layers.experimental.preprocessing.Rescaling(1./225),
    layers.experimental.preprocessing.RandomContrast(0.5, seed = 123)
])

inputs = Input(shape = (150,150,3))
x = resize_and_rescale(inputs)
x = data_augmentation(x)
x = layers.Conv2D(64, 3,   activation = 'relu',bias_regularizer=L2(0.001), padding = 'same', kernel_regularizer=L2(0.001))(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)
x = layers.BatchNormalization(momentum = 0.1)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, 3,  activation = 'relu',bias_regularizer=L2(0.001), padding = 'same',kernel_regularizer=L2(0.001))(x)
x = layers.BatchNormalization(momentum = 0.1)(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)
x = layers.Conv2D(256, 3,   activation = 'relu',bias_regularizer=L2(0.001), padding = 'same',kernel_regularizer=L2(0.001))(x)
x = layers.BatchNormalization(momentum = 0.1)(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)
x = layers.Conv2D(512, 3,  activation = 'relu',bias_regularizer=L2(0.001), padding = 'same',kernel_regularizer=L2(0.001))(x)
x = layers.BatchNormalization(momentum = 0.1)(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)
x = layers.Conv2D(256, 3,  activation = 'relu',bias_regularizer=L2(0.001), padding = 'same',kernel_regularizer=L2(0.001))(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation = 'relu')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(inputs=inputs, outputs = outputs)

print(model.summary())

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(0.001),
              metrics = ['accuracy'])

train_dir = '/home/eirikur/deep/project/cats_and_dogs_small/train/'
validation_dir = '/home/eirikur/deep/project/cats_and_dogs_small/validation/'
test_dir = '/home/eirikur/deep/project/cats_and_dogs_small/test/'

train_datagen = ImageDataGenerator(rotation_range= 40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   fill_mode = 'nearest',
                                   horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'binary'
)
history = model.fit(train_generator,
                    steps_per_epoch = 2000//32,
                    epochs = 30,
                    validation_data = validation_generator,
                    validation_steps = 500//32)

results = model.evaluate(test_generator, batch_size = 32)
print(results)
model.save('cats_and_dogs_small.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.legend()

plt.show()
