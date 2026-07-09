# -*- coding: utf-8 -*-

# import the libraries as shown below
import matplotlib

from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
matplotlib.use('agg')  # Set the backend to 'agg' before importing pyplot
# re-size all the images to this
import matplotlib.pyplot as plt
IMAGE_SIZE = [224, 224]
targetsize = (224, 224)
train_path = 'dataset/train'
valid_path = 'dataset/test'

# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# useful for getting number of output classes
folders = glob('dataset/train/*')

print("Dataset Loaded")
# our layers - you can add more if you want
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Make sure you provide the same target size as initialized for the image size
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=targetsize,
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size=targetsize,
                                            batch_size=32,
                                            class_mode='categorical')
print(test_set)

# Add checkpoint to save the best model during training
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('alzimerstemp.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=[checkpoint]
)

# Save the model after training
#model.save('tomato.h5')

# plot the loss
'''plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss')
plt.show()

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc')
plt.show()'''
plt.figure(figsize=(10, 6))

# Plot Loss
plt.plot(r.history['loss'], label='Train Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')

# Plot Accuracy
plt.plot(r.history['accuracy'], label='Train Accuracy')
plt.plot(r.history['val_accuracy'], label='Validation Accuracy')

plt.title('Training & Validation Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('combined_loss_accuracy.png')
plt.show()

