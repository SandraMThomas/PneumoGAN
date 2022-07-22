from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sys
import numpy as np

# Input shape
imageRows = 128
imageCols = 128
channels = 1
imageShape = (imageRows, imageCols, channels)
latentDimension = 100

myXrayOptimizer = Adam(0.0001, 0.8)
global discAcc
discAcc = []


def myGenerator(latentDimension1):

    XrayGenModel = Sequential()

    XrayGenModel.add(Dense(1024 * 4 * 4, activation="relu",
                     input_dim=latentDimension1))
    XrayGenModel.add(Reshape((4, 4, 1024)))
    XrayGenModel.add(UpSampling2D())
    XrayGenModel.add(Conv2DTranspose(512, kernel_size=3,
                     strides=(1, 1), dilation_rate=2, padding="same"))
    XrayGenModel.add(BatchNormalization(momentum=0.8))
    XrayGenModel.add(Activation("relu"))
    XrayGenModel.add(UpSampling2D())
    XrayGenModel.add(Conv2DTranspose(256, kernel_size=3,
                     strides=(1, 1), dilation_rate=2, padding="same"))
    XrayGenModel.add(BatchNormalization(momentum=0.8))
    XrayGenModel.add(Activation("relu"))
    XrayGenModel.add(UpSampling2D())
    XrayGenModel.add(Conv2DTranspose(128, kernel_size=3,
                     strides=(1, 1), dilation_rate=2, padding="same"))
    XrayGenModel.add(BatchNormalization(momentum=0.8))
    XrayGenModel.add(Activation("relu"))
    XrayGenModel.add(UpSampling2D())
    XrayGenModel.add(Conv2DTranspose(64, kernel_size=3,
                     strides=(1, 1), dilation_rate=2, padding="same"))
    XrayGenModel.add(BatchNormalization(momentum=0.8))
    XrayGenModel.add(Activation("relu"))
    XrayGenModel.add(UpSampling2D())
    XrayGenModel.add(Conv2DTranspose(128, kernel_size=3,
                     strides=(1, 1), dilation_rate=2, padding="same"))
    XrayGenModel.add(BatchNormalization(momentum=0.8))
    XrayGenModel.add(Activation("relu"))

    XrayGenModel.add(Conv2DTranspose(1, kernel_size=3,
                     strides=(1, 1), dilation_rate=2, padding="same"))
    XrayGenModel.add(Activation("tanh"))

    XrayGenModel.summary()

    latentNoise = Input(shape=(latentDimension,))
    imgGenerated = XrayGenModel(latentNoise)

    return Model(latentNoise, imgGenerated)


def myDiscriminator(imageShape1):

    XrayDiscModel = Sequential()

    XrayDiscModel.add(Conv2D(32, kernel_size=3, strides=(
        2, 2), input_shape=imageShape1, padding="same"))
    XrayDiscModel.add(LeakyReLU(alpha=0.2))
    XrayDiscModel.add(Dropout(0.25))
    XrayDiscModel.add(
        Conv2D(64, kernel_size=3, strides=(2, 2), padding="same"))
    XrayDiscModel.add(BatchNormalization(momentum=0.8))
    XrayDiscModel.add(LeakyReLU(alpha=0.2))
    XrayDiscModel.add(Dropout(0.25))
    XrayDiscModel.add(
        Conv2D(128, kernel_size=3, strides=(2, 2), padding="same"))
    XrayDiscModel.add(BatchNormalization(momentum=0.8))
    XrayDiscModel.add(LeakyReLU(alpha=0.2))
    XrayDiscModel.add(Dropout(0.25))
    XrayDiscModel.add(
        Conv2D(256, kernel_size=3, strides=(2, 2), padding="same"))
    XrayDiscModel.add(BatchNormalization(momentum=0.8))
    XrayDiscModel.add(LeakyReLU(alpha=0.2))
    XrayDiscModel.add(Dropout(0.25))
    XrayDiscModel.add(
        Conv2D(512, kernel_size=3, strides=(2, 2), padding="same"))
    XrayDiscModel.add(BatchNormalization(momentum=0.8))
    XrayDiscModel.add(LeakyReLU(alpha=0.2))
    XrayDiscModel.add(Dropout(0.25))
    XrayDiscModel.add(Flatten())
    XrayDiscModel.add(Dense(1, activation='sigmoid'))

    XrayDiscModel.summary()

    imgToBeDiscriminated = Input(shape=imageShape1)
    validityDiscriminator = XrayDiscModel(imgToBeDiscriminated)

    return Model(imgToBeDiscriminated, validityDiscriminator)


global trainSet
# the discriminator(1.building 2.compling)
XrayDiscriminator = myDiscriminator(imageShape)
XrayDiscriminator.compile(loss='binary_crossentropy',
                          optimizer=myXrayOptimizer, metrics=['accuracy'])

# the generator(1.building)
XrayGenerator = myGenerator(latentDimension)

# The generator takes noise as input and generates imgs
latentVector = Input(shape=(latentDimension,))
img = XrayGenerator(latentVector)

XrayDiscriminator.trainable = True

# The discriminator takes generated images as input and determines validity
valid = XrayDiscriminator(img)

# The GAN model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
myGAN = Model(latentVector, valid)
myGAN.compile(loss='binary_crossentropy', optimizer=myXrayOptimizer)


def myXrayGAN(epochs=100, batch_size=32):
    global trainSet, discAcc
    discAcc = []
    (imgX, imgY) = 128, 128
    trainPath = "D://S8//Project//fiNAL//resources//other_files//newDataTrain.csv"

    # , 'No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax']
    className = ['Atelectasis']
    numClasses = 5

    # Load training data
    dataTrain = pd.read_csv(trainPath)

    xTrain = []
    # prepare label binarizer
    imagePath = "D://S8//Project//fiNAL//dataset//training_data//train//"

    count = 0
    for index, row in dataTrain[dataTrain["Finding Labels"] == 'Pneumothorax'].iterrows():
        imgName = imagePath + row["Image Index"]
        imageRead = cv2.imread(imgName)  # Image.open(img).convert('L')
        imageRead = imageRead[:, :, 0]
        imageArray = cv2.resize(imageRead, (imgX, imgY))
        imageArray = imageArray.astype('float32')
        imageArray /= 255.0
        imageArray = imageArray - np.mean(imageArray)
        xTrain.append(imageArray)
        count += 1

    print("shape of x train: {}".format(len(xTrain)))
    xTrain = np.asarray(xTrain)

    xTrain = xTrain.reshape(count, imgY, imgX, 1)

    validCheck = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    trainSet = xTrain
    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half of images
        idX = np.random.randint(0, xTrain.shape[0], batch_size)
        imgSelected = xTrain[idX]

        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latentDimension))
        generatedImgs = XrayGenerator.predict(noise)

        # Train the discriminator (real classified as ones and generated as zeros)
        dLossReal = XrayDiscriminator.train_on_batch(imgSelected, validCheck)
        dLossFake = XrayDiscriminator.train_on_batch(generatedImgs, fake)
        discriminatorLoss = 0.5 * np.add(dLossReal, dLossFake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (wants discriminator to mistake images as real)
        generatorLoss = myGAN.train_on_batch(noise, validCheck)

        # Plot the progress
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
            epoch, discriminatorLoss[0], 100 * discriminatorLoss[1], generatorLoss))
        discAcc.append(100 * discriminatorLoss[1])
        
        #save_imgs(epoch)
# def save_imgs(epoch):
#     r, c = 100, 10
#     noise = np.random.normal(0, 1, (r * c, latentDimension))
#     gen_imgs =XrayGenerator.predict(noise)

#     # Rescale images 0 - 1
#     gen_imgs = 0.5 * gen_imgs + 0.5

#     fig, axs = plt.subplots(r, c)
#     cnt = 0
#     for i in range(r):
#         for j in range(c):
#             axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
#             axs[i, j].axis('off')
#             cnt += 1
#     fig.savefig("images\\xrays_%d.png" % epoch)
#     plt.close()

if __name__ == '__main__':
    epochs = 20
    myXrayGAN(epochs=epochs, batch_size=32)
    epochs = [i for i in range(epochs)]
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 10)
    ax.plot(epochs, discAcc, 'go-', label=' Accuracy')
    ax.set_title(' Accuracy ')
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    plt.show()
