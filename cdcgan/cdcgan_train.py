import matplotlib

matplotlib.use('Agg')
import numpy as np

from tqdm import tqdm
from keras.datasets import cifar10
from keras.optimizers import Adam
import math

from cdcgan.dcgan import DCGAN
from utils.utils import TextImage2013
from recognition.recognition import Recognizer

BATCH_SIZE = 9
EPOCHS = 1
RANDOM_SIZE = 100

from random import randrange
from PIL import Image, ImageDraw, ImageFont
import textwrap

def draw_text(imgdim, text, position, color3, fontsize, width_wrap):

    data = np.zeros(imgdim, dtype=np.uint8)
    image = Image.fromarray(data, 'RGB')

    draw = ImageDraw.Draw(image)

    font_path = '../recognition/Arial-Unicode-Regular.ttf'
    font = ImageFont.truetype(font_path, size=fontsize)

    color = 'rgb(%d, %d, %d)'%(color3[0], color3[1], color3[2])

    lines = textwrap.wrap(text, width=width_wrap)

    y_text = position[0]
    for line in lines:
        width, height = font.getsize(line)
        draw.text((position[1], y_text), line, fill=color, font=font)
        y_text += height

    return np.array(image)

def prepare_conditions(batch, imgdim):
    answ = []

    for i in range(0, batch):
        position = (randrange(200), randrange(200))
        color3 = (randrange(255), randrange(255), randrange(255))
        fonsize = 20 + randrange(100)
        text = chr(ord('a') + (i % 27)) + chr(ord('b') + (i % 25))

        img = draw_text(imgdim, text, position, color3, fonsize, 100)

        answ.append(img)

    return np.array(answ)

def save_imagegrid(imagearray, img_name, batch_size):
    width = imagearray.shape[2]
    height = imagearray.shape[1]
    mode = 'RGB'
    num_elements = int(math.sqrt(batch_size))
    imagegrid = Image.new(mode, (width * num_elements, height * num_elements))
    for j in range(num_elements * num_elements):
        randimg = imagearray[j] * 127.5 + 127.5
        img = Image.fromarray(randimg.astype('uint8'), mode=mode)
        imagegrid.paste(im=img, box=((j % num_elements) * width, height * (j // num_elements)))
    filename = str(img_name) + '.png'
    imagegrid.save(filename)
    print("grid saved " + filename)

def recognition_loss(y_true, y_pred):
    print('aaa')
    return 1.0

def loss(y_true, y_pred):
    loss = 1.0
    return loss + recognition_loss(y_true, y_pred)

#(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

training = TextImage2013()

xtrain = training.get_images()
ytrain = training.get_labels()

xtrain = xtrain.astype(np.float32)
xtrain = (xtrain - 127.5) / 127.5

print(xtrain.shape)
imgdim = (512, 512, 3)

# Create the models

Ganmodel = DCGAN(img_dims=xtrain.shape[1:], gt=training)
generator = Ganmodel.generator512()
generator.summary()
gen_sgd = Adam(lr=0.0002, beta_1=0.5)
generator.compile(loss=loss, optimizer='sgd') # todo: add recognition loss

discriminator = Ganmodel.discriminator512()
discriminator.summary()
discriminator_sgd = Adam(lr=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_sgd)

gan = Ganmodel.generator_containing_discriminator(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=gen_sgd)

# Model Training

iteration = 0

nb_of_iterations_per_epoch = int(xtrain.shape[0] / BATCH_SIZE)
print("Number of iterations per epoch: {0}".format(nb_of_iterations_per_epoch))
losses = {"generator": [], "discriminator": []}

for epoch in range(EPOCHS):
    pbar = tqdm(desc="Epoch: {0}".format(epoch), total=xtrain.shape[0])

    for i in range(nb_of_iterations_per_epoch):
        random_array = np.random.uniform(-1, 1, (BATCH_SIZE, RANDOM_SIZE)) #(128, 100)

        conditions = prepare_conditions(BATCH_SIZE, imgdim)

        generated_images = generator.predict_on_batch([random_array, conditions])

        if ((i % 10) == 0):
            save_imagegrid(generated_images, '../images/generated_per_epoch/e{:02d}b{:03d}'.format(epoch, i), BATCH_SIZE)

        if (i % 2 == 0):
            start = (i // 2 * BATCH_SIZE)
            end = (((i // 2) + 1) * BATCH_SIZE)
            images = xtrain[start : end, :, :, :]
            labels = [1] * (BATCH_SIZE)
        else:
            images = generated_images
            labels = [0] * (BATCH_SIZE)

        discriminator_loss = discriminator.train_on_batch(images, labels)
        losses["discriminator"].append(discriminator_loss)

        random_array = np.random.uniform(-1, 1, (BATCH_SIZE, RANDOM_SIZE))
        conditions = prepare_conditions(BATCH_SIZE, imgdim)

        labels = [1] * (BATCH_SIZE)
        discriminator.trainable = False
        generator_loss = gan.train_on_batch([random_array, conditions], labels)
        discriminator.trainable = True
        losses["generator"].append(generator_loss)

        pbar.update(BATCH_SIZE)

        iteration += 1

        discriminator.save_weights('../models/discriminator_cifar.h5')
        generator.save_weights('../models/generator_cifar.h5')

pbar.close()

def generate():
    dims = (32, 32, 3)
    gan = DCGAN(img_dims=dims)
    generator = gan.generator512()
    generator.load_weights('generator.h5')
    generator.compile(loss='binary_crossentropy', optimizer='sgd')
    random_array = np.random.uniform(-1, 1, (BATCH_SIZE, RANDOM_SIZE))
    conditions = prepare_conditions(BATCH_SIZE, imgdim)
    generated_images = generator.predict_on_batch([random_array, conditions])
    save_imagegrid(generated_images, 'generated_images', BATCH_SIZE)
