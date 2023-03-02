from keras.models import Sequential, Model
from keras.layers import Activation, Lambda, Input, Add, Subtract
from keras.layers.core import Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal


def transposed_conv(model, out_channels):
    model.add(Conv2DTranspose(3, (5, 5), activation="relu", strides=(2, 2), padding="same", kernel_initializer="uniform"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    return model

def conv(model, out_channels):
    model.add(Conv2D(out_channels, (5, 5),
                     kernel_initializer=RandomNormal(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    return model

def add(x):
    return x[0] + x[1]

class DCGAN():

    def __init__(self, img_dims):
        self.imwidth = img_dims[0]
        self.imheight = img_dims[1]
        self.imchannels = img_dims[2]
        self.downsize_factor = 3
        self.random_array_size = 100

    # def generator(self):
    #     scale = 2 ** self.downsize_factor
    #     model = Sequential()
    #     k = self.imwidth // scale * self.imheight // scale * 1024
    #     model.add(Dense(k, input_dim=self.random_array_size, kernel_initializer=RandomNormal(stddev=0.02)))
    #     model.add(BatchNormalization())
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Reshape([self.imheight // scale, self.imwidth // scale, 1024]))
    #     model = transposed_conv(model, 64)
    #     model = transposed_conv(model, 32)
    #     model.add(Conv2DTranspose(self.imchannels, [5, 5], strides=(2, 2),
    #         activation='tanh', padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
    #     model.add(Lambda(draw_text, output_shape=(self.imwidth, self.imheight, self.imchannels))) #,
    #                      #arguments={"text": text, "position": ..., "color": ..., "fontsize": ..., "widtg_wrap": ...}))
    #     # model.add(Conv2DTranspose(self.imchannels, [5, 5], strides=(2, 2),
    #     #                           activation='tanh', padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
    #     # model.add(Conv2DTranspose(self.imchannels, [5, 5], strides=(2, 2),
    #     #                           activation='tanh', padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
    #     return model

    def generator32(self):
        scale = 2 ** self.downsize_factor
        k = self.imwidth // scale * self.imheight // scale * 1024

        # 100
        inputs = Input((self.random_array_size,), name="input_0")

        l1 = BatchNormalization(name="bn1")(inputs)

        # 16384
        l2 = Dense(k, activation='relu', name="fc1")(l1)
        l3 = BatchNormalization(name="bn2")(l2)
        l4 = LeakyReLU(alpha=0.2, name="relu1")(l3)

        # 4 x 4 x 1024
        l5 = Reshape([self.imheight // scale, self.imwidth // scale, 1024], name="reshape")(l4)

        # Prepare Conditional input
        input_c = Input((self.imwidth, self.imheight, self.imchannels), name="input_c")

        # 8 x 8 x 3
        l6 = Conv2DTranspose(3, (5, 5), activation="relu", strides=(2, 2), padding="same", kernel_initializer="uniform", name="conv_transpose_1")(l5)
        l7 = LeakyReLU(alpha=0.2, name="relu2")(l6)

        # 16 x 16 x 3
        l8 = Conv2DTranspose(3, (5, 5), activation="relu", strides=(2, 2), padding="same", kernel_initializer="uniform", name="conv_transpose_2")(l7)
        l9 = LeakyReLU(alpha=0.2, name="relu3")(l8)

        # 32 x 32 x 3
        l10 = Conv2DTranspose(self.imchannels, [5, 5], strides=(2, 2), activation='tanh',
                              padding='same', kernel_initializer=RandomNormal(stddev=0.02), name="conv_transpose_3")(l9)

        # 32 x 32 x 3
        l11 = Lambda(add, output_shape=(self.imwidth, self.imheight, self.imchannels), name="add_text")([l10, input_c])

        l12 = Conv2DTranspose(3, [5, 5], strides=(1, 1), activation='tanh',
                              padding='same', kernel_initializer=RandomNormal(stddev=0.02), name="conv_transpose_4")(l11)
        l13 = LeakyReLU(alpha=0.2, name="relu4")(l12)

        l14 = Conv2DTranspose(3, [5, 5], strides=(1, 1), activation='tanh',
                              padding='same', kernel_initializer=RandomNormal(stddev=0.02), name="conv_transpose_5")(l13)
        l15 = LeakyReLU(alpha=0.2, name="relu5")(l14)

        # add some layers to process text

        model = Model(input=[inputs, input_c], output=l15)
        return model

    def discriminator32(self):
        downsize = self.downsize_factor
        model = Sequential()
        model.add(Conv2D(64, (5, 5), input_shape=(
            self.imheight, self.imwidth, self.imchannels), kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model = conv(model, 128)
        model.add(AveragePooling2D(pool_size=(2, 2)))
        if (downsize == 3):
            model = conv(model, 128)
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def generator256(self):
        scale = 2 ** self.downsize_factor
        k = self.imwidth // scale * self.imheight // scale * 1024

        # 100
        inputs = Input((self.random_array_size,), name="input_0")

        l1 = BatchNormalization(name="bn1")(inputs)

        # 105906176
        l2 = Dense(k, activation='relu', name="fc1")(l1)
        l3 = BatchNormalization(name="bn2")(l2)
        l4 = LeakyReLU(alpha=0.2, name="relu1")(l3)

        # 32 x 32 x 1024
        l5 = Reshape([self.imheight // scale, self.imwidth // scale, 1024], name="reshape")(l4)

        # Prepare Conditional input
        input_c = Input((self.imwidth, self.imheight, self.imchannels), name="input_c")

        # 64 x 64 x 3
        l6 = Conv2DTranspose(3, (5, 5), activation="relu", strides=(2, 2), padding="same", kernel_initializer="uniform", name="conv_transpose_1")(l5)
        l7 = LeakyReLU(alpha=0.2, name="relu2")(l6)

        # 128 x 128 x 3
        l8 = Conv2DTranspose(3, (5, 5), activation="relu", strides=(2, 2), padding="same", kernel_initializer="uniform", name="conv_transpose_2")(l7)
        l9 = LeakyReLU(alpha=0.2, name="relu3")(l8)

        # 256 x 256 x 3
        l10 = Conv2DTranspose(self.imchannels, [5, 5], strides=(2, 2), activation='tanh',
                              padding='same', kernel_initializer=RandomNormal(stddev=0.02), name="conv_transpose_3")(l9)
        l105 = LeakyReLU(alpha=0.2, name="relu4")(l10)

        # 256 x 256 x 3
        #l11 = Lambda(add, output_shape=(self.imwidth, self.imheight, self.imchannels), name="add_text")([l10, input_c])
        l11 = Add(name="add_text")([l105, input_c])

        # layers to process text

        # # 512 x 512 x 3
        # l12 = Conv2DTranspose(3, [5, 5], strides=(1, 1), activation='tanh',
        #                       padding='same', kernel_initializer=RandomNormal(stddev=0.02), name="conv_transpose_4")(l11)
        # l13 = LeakyReLU(alpha=0.2, name="relu5")(l12)
        #
        # # 512 x 512 x 3
        # l14 = Conv2DTranspose(3, [5, 5], strides=(1, 1), activation='tanh',
        #                       padding='same', kernel_initializer=RandomNormal(stddev=0.02), name="conv_transpose_5")(l13)
        # l15 = LeakyReLU(alpha=0.2, name="relu6")(l14)

        # recornition

        r1 = Dense(128, activation="relu", name="fcr1")(inputs)

        r2 = LeakyReLU(alpha=0.2, name="relur")(r1)

        model = Model(input=[inputs, input_c], output=[l11, r2]) #l15)
        return model

    def discriminator256(self):

        # 256 x 256 x 3
        inputs = Input((self.imwidth, self.imheight, self.imchannels), name="input_d")

        # 252 x 252 x 64
        l1 = Conv2D(64, (5, 5), input_shape=(self.imheight, self.imwidth, self.imchannels),
                         kernel_initializer=RandomNormal(stddev=0.02), name="conv1")(inputs)
        l2 = LeakyReLU(alpha=0.2, name="relu1")(l1)

        # 126 x 126 x 64
        l3 = AveragePooling2D(pool_size=(2, 2), name="pool1")(l2)

        # 122 x 122 x 128
        l4 = Conv2D(128, (5, 5), kernel_initializer=RandomNormal(stddev=0.02), name="conv2")(l3)
        l5 = BatchNormalization(name="bn1")(l4)
        l6 = LeakyReLU(alpha=0.2, name="relu2")(l5)

        # 61 x 61 x 128
        l7 = AveragePooling2D(pool_size=(2, 2), name="pool2")(l6)

        # 57 x 57 x 128
        l8 = Conv2D(128, (5, 5), kernel_initializer=RandomNormal(stddev=0.02), name="conv3")(l7)
        l9 = BatchNormalization(name="bn2")(l8)
        l10 = LeakyReLU(alpha=0.2, name="rely3")(l9)

        # 415872
        l11 = Flatten(name="flatten")(l10)
        l12 = Dense(1, name="dense")(l11)
        l13 = Activation('sigmoid', name="sigmoid")(l12)

        model = Model(input=inputs, output=l13)

        return model


    def generator_containing_discriminator(self, g, d):
        input_z = Input((self.random_array_size,))
        input_c = Input((self.imwidth, self.imheight, self.imchannels))
        gen_image, texts = g([input_z, input_c])
        d.trainable = False
        is_real = d(gen_image)
        model = Model(inputs=[input_z, input_c], outputs=[is_real, texts])
        return model