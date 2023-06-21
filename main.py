import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten, Input,
                                     LeakyReLU, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def Build(width, height, depth, filters=(32, 64), latentDim=16):
    # Encoder
    inputShape = (height, width, depth)
    chanDim = -1

    inputs = Input(shape=inputShape)
    x = inputs

    for f in filters:
        x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    volumeSize = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latentDim)(x)
    encoder = Model(inputs, latent, name="encoder")

    # Decoder
    latentInputs = Input(shape=(latentDim,))
    x = Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    for f in filters[::-1]:
        x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
    outputs = Activation("sigmoid")(x)
    decoder = Model(latentInputs, outputs, name="decoder")

    # Autoencoder
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

    return encoder, decoder, autoencoder


def main() -> int:
    encoder, decoder, autoencoder = Build(28, 28, 1)

    EPOCHS = 25
    BS = 32

    ((trainX, _), (testX, _)) = mnist.load_data()

    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    opt = Adam(lr=1e-3)
    autoencoder.compile(loss="mse", optimizer=opt)

    H = autoencoder.fit(
        trainX, trainX, validation_data=(testX, testX), epochs=EPOCHS, batch_size=BS
    )

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
