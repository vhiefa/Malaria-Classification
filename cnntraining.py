from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    directory='./dataset/train_set',
    target_size=(100,100),
    color_mode='rgb',
    batch_size=32,
    #seed = 42
    shuffle=True,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    directory='./dataset/test_set',
    target_size=(100,100),
    color_mode='rgb',
    batch_size=32,
    shuffle=False,
    #seed = 42
    class_mode='binary'
)

#initialize the CNN
classifier = Sequential()
classifier.add(Convolution2D(32, (3,3),strides=2, input_shape=(100,100,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides=2))
classifier.add(Convolution2D(32, (3,3),strides=2, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides=2))
classifier.add(Convolution2D(64, (3,3),strides=2, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides=2))
classifier.add(Flatten())
classifier.add(Dense(output_dim=64, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))
#classifier.add(Dense(output_dim=2, activation='softmax'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])

STEP_SIZE_TRAIN= training_set.n / 10#//training_set.batch_size
STEP_SIZE_TEST= test_set.n / 10#//test_set.batch_size
NUMBER_OF_EPOCH = 50

#TRAINING
H = classifier.fit_generator(
    generator=training_set,
    steps_per_epoch=STEP_SIZE_TRAIN, #1,#8000
    epochs=NUMBER_OF_EPOCH,
    validation_data=test_set,
    validation_steps=STEP_SIZE_TEST#1,#800
)

# serialize model to JSON
model_json = classifier.to_json()
with open("model_nfl_fnl.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model_nfl_fnl.h5')
print('Model has saved to disk!')

# Plot Result
plt.style.use("ggplot")
plt.figure()

#Plot Loss
plt.subplot(311)
plt.title("Training on Dataset")
plt.plot(np.arange(0, NUMBER_OF_EPOCH), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, NUMBER_OF_EPOCH), H.history["val_loss"], label="test_loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

#Plot Acc
plt.subplot(312)
plt.legend(loc="lower left")
plt.plot(np.arange(0, NUMBER_OF_EPOCH), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, NUMBER_OF_EPOCH), H.history["val_acc"], label="test_acc")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

#Plot MSE
plt.subplot(313)
plt.plot(np.arange(0, NUMBER_OF_EPOCH), H.history["mean_squared_error"], label="train_mse")
plt.plot(np.arange(0, NUMBER_OF_EPOCH), H.history["val_mean_squared_error"], label="test_mse")
plt.ylabel("Mean Squared Error")
plt.xlabel("Epoch #")
plt.legend(loc="lower left")

plt.savefig("plot.png")
plt.show()

