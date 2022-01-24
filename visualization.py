#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
image_path = '/home/eirikur/deep/project/cats_and_dogs_small/test/cats/cat.1250.jpg'

model_path = '/home/eirikur/deep/project/scripts/cats_and_dogs_Classifier.h5'

model = models.load_model(model_path)

img = image.load_img(image_path, target_size = (150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor /= 255.

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(img_tensor)

first_activation = activations[0]

plt.matshow(first_activation[0,:,:,0], cmap = 'viridis')
plt.matshow(first_activation[0,:,:,1], cmap = 'viridis')
plt.matshow(first_activation[0,:,:,3], cmap = 'viridis')
plt.matshow(first_activation[0,:,:,4], cmap = 'viridis')
plt.matshow(first_activation[0,:,:,5], cmap = 'viridis')
plt.matshow(first_activation[0,:,:,30], cmap = 'viridis')
plt.matshow(first_activation[0,:,:,42], cmap = 'viridis')
plt.show()

print(first_activation.shape)
print(type(first_activation))
