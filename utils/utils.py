import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

class TextImage2013():

    def __init__(self):
        self.image_path = '../../data/ICDAR/end-to-end/ch2_training_images/'
        self.gt_path = '../../data/ICDAR/end-to-end/ch2_training_localization_transcription_gt/'
        self.orig_shape = {}
        self.ytrain = []
        self.w = 512
        self.h = 512
        self.ch = 3
        self.nitems = 229
        self.xtrain = np.ndarray(shape=(self.nitems, self.h, self.w, self.ch))

        for (i, path) in enumerate(os.listdir(self.image_path)):
            original = img_to_array(load_img(self.image_path + path))
            self.orig_shape[i] = original.shape
            resized = original.resize((self.h, self.w))
            self.xtrain[i] = resized

        print(self.orig_shape)

        for (i, path) in enumerate(os.listdir(self.gt_path)):
            l = self.read_gt_data(self.gt_path + path, self.orig_shape[i])
            self.ytrain.append(l)

    def read_gt_data(self, path, orsh):
        tb = Textbox()
        tb.read_from_file(path, orsh)
        return tb

    def get_images(self):
        return self.xtrain

    def get_labels(self):
        return self.ytrain

class Textbox():
    def __init__(self):
        self.corners = []
        self.resized_corners = []
        self.text = []
        self.nboxes = 0

    def read_from_file(self, path, orsh):
        f = open(path, "r")
        nh, nw, nc = orsh
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
        print(self.nboxes)
        print(self.text)
        print(self.corners)