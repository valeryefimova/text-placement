import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import cv2

class TextImage2013():

    def __init__(self):
        self.image_folder_path = '/media/disk2/vefimova/icdar/ch2_training_images/'
        self.gt_path = '/media/disk2/vefimova/icdar/ch2_training_localization_transcription_gt/'
        self.image_path = []
        self.labels = []
        self.w = 256
        self.h = 256
        self.ch = 3

        for (i, path) in enumerate(os.listdir(self.image_folder_path)):
            print(path + " " + str(i + 1))
            self.image_path.append(self.image_folder_path + path)
            l = self.read_gt_data(self.gt_path + "gt_" + path[:-4] + ".txt")
            self.labels.append(l)

        self.num_samples = len(self.image_path)

    def read_gt_data(self, path):
        tb = Textbox()
        tb.read_from_file(path)
        return tb

    def get_labels(self):
        return self.labels

    def image_generator(self, batch_size):
        idxs = np.random.randint(0, self.num_samples, batch_size)

        inputs = []

        for i in idxs:
            # original = img_to_array(load_img(self.image_path[i]))
            # resized = np.asarray(original.resize((self.h, self.w)), dtype=np.float32)
            original = cv2.imread(self.image_path[i])
            resized = cv2.resize(original, dsize=(self.h, self.w))
            resized.astype(np.float32)
            resized = (resized - 127.5) / 127.5
            inputs.append(resized)

        return np.asarray(inputs)


class Textbox():
    def __init__(self):
        self.corners = []
        self.resized_corners = []
        self.text = []
        self.nboxes = 0

    def read_from_file(self, path):
        f = open(path, "r", encoding='utf-8')
        for box in f:
            box = box.rstrip()
            all = box.split(',')
            if (all[8] == '###'):
                continue
            self.text.append(all[8])
            all[0] = all[0][1:]
            self.corners.append([(int(x) if (x != '') else '') for x in all[:8]])
            self.nboxes += 1
        f.close()
        #print(self.nboxes)
        #print(self.text)
        #print(self.corners)

from random import randrange
from PIL import Image, ImageDraw, ImageFont
import textwrap
from scipy.misc import imresize
from keras.datasets import cifar10

(source, l), (_, _) = cifar10.load_data()

def draw_text(imgdim, text, position, color3, fontsize, width_wrap):

    data = np.zeros(imgdim, dtype=np.uint8)
    #ind = np.random.randint(source.shape[0])
    #data = source[ind]
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

    image = np.array(image, dtype=np.float32)
    image = (image - 127.5) / 127.5
    image = imresize(image, (imgdim[0], imgdim[1]))

    return image

def prepare_conditions(batch_size, imgdim):
    answ = []
    texts = []

    for i in range(0, batch_size):
        position = (randrange(150), randrange(150))
        color3 = (randrange(255), randrange(255), randrange(255))
        fonsize = 10 + randrange(80)
        text = chr(ord('a') + (i % 27)) + chr(ord('b') + (i % 25))
        texts.append(text)

        img = draw_text(imgdim, text, position, color3, fonsize, 100)

        answ.append(img)

    return texts, np.array(answ)