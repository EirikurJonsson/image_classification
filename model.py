#!/usr/bin/env python3

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = models.Sequential()

# create the first model
model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape = (150,150,3)))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.BatchNormalization(momentum = 0.1))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=L2(0.001), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(256, (3,3), padding = 'same', kernel_regularizer=L2(0.001), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.BatchNormalization(momentum = 0.1))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(learning_rate= 0.0001),
              metrics = ['accuracy'])

train_dir = '/home/eirikur/deep/project/cats_and_dogs_small/train/'
validation_dir = '/home/eirikur/deep/project/cats_and_dogs_small/validation/'
test_dir = '/home/eirikur/deep/project/cats_and_dogs_small/test/'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range= 40,
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
    batch_size = 20,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)
history = model.fit(train_generator,
                    steps_per_epoch = 100,
                    epochs = 100,
                    validation_data = validation_generator,
                    validation_steps = 20)

results = model.evaluate(test_generator, batch_size = 20)
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
