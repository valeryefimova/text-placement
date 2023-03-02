import cv2
import numpy as np
from matplotlib import pyplot as plt

from recognition.nms import get_boxes

from recognition.models import ModelResNetSep2
import recognition.net_utils as net_utils

from recognition.ocr_utils import ocr_image
from recognition.data_gen import draw_box_points

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class Recognizer:


    def __init__(self):

        f = open('recognition/codec.txt', 'r', encoding='utf-8')
        self.codec = f.readlines()[0]
        f.close()

        self.net = ModelResNetSep2(attention=True)
        net_utils.load_net("recognition/e2e-mlt.h5", self.net)
        self.net = self.net.eval()

        self.font2 = ImageFont.truetype("recognition/Arial-Unicode-Regular.ttf", 18)

    def resize_image(self, img, max_size = 1585152, scale_up=True):

        if scale_up:
            image_size = [img.shape[1] * 6 // 32 * 32, img.shape[0] * 6 // 32 * 32]
        else:
            image_size = [img.shape[1] // 32 * 32, img.shape[0] // 32 * 32]
        while image_size[0] * image_size[1] > max_size:
            image_size[0] /= 1.2
            image_size[1] /= 1.2
            image_size[0] = int(image_size[0] // 32) * 32
            image_size[1] = int(image_size[1] // 32) * 32

        resize_h = int(image_size[1])
        resize_w = int(image_size[0])

        scaled = cv2.resize(img, dsize=(resize_w, resize_h))
        return scaled, (resize_h, resize_w)

    def recognize_image(self, im):
        im_resized, (ratio_h, ratio_w) = self.resize_image(im, scale_up=False)
        images = np.asarray([im_resized], dtype=np.float)
        images /= 128
        images -= 1
        im_data = net_utils.np_to_variable(images, is_cuda=False).permute(0, 3, 1, 2)
        seg_pred, rboxs, angle_pred, features = self.net(im_data)

        rbox = rboxs[0].data.cpu()[0].numpy()
        rbox = rbox.swapaxes(0, 1)
        rbox = rbox.swapaxes(1, 2)

        angle_pred = angle_pred[0].data.cpu()[0].numpy()

        segm = seg_pred[0].data.cpu()[0].numpy()
        segm = segm.squeeze(0)

        boxes =  get_boxes(segm, rbox, angle_pred, False)

        eps = 100.0

        if len(boxes) > 10:
            boxes = boxes[0:10]

        out_boxes = []
        out_text = []
        for box in boxes:

            probability = box[8]
            if (probability < eps):
                continue

            pts = box[0:8]
            pts = pts.reshape(4, -1)

            det_text, conf, dec_s = ocr_image(self.net, self.codec, im_data, box)
            out_text.append(det_text)
            if len(det_text) == 0:
                continue

            out_boxes.append(box)
            print(det_text)


        # im = np.array(img)
        # plt.imshow(im)
        # plt.show()

        return out_text, out_boxes

    def draw(self, im, texts, boxes):
        im_resized, (ratio_h, ratio_w) = self.resize_image(im, scale_up=False)
        draw2 = np.copy(im_resized)
        img = Image.fromarray(draw2)
        draw = ImageDraw.Draw(img)

        for i in range(0, len(boxes)):
            box = boxes[i]
            text = texts[i]
            pts  = box[0:8]
            pts = pts.reshape(4, -1)
            draw_box_points(im, pts, color=(0, 255, 0), thickness=1)

            width, height = draw.textsize(text, font=self.font2)
            center = [box[0], box[1]]
            draw.text((center[0], center[1]), text, fill=(0, 255, 0), font=self.font2)

        im = np.array(img)
        plt.imshow(im)
        plt.show()

# rec = Recognizer()
# for i in range(1, 10):
#     img = cv2.imread("../../data/ICDAR/end-to-end/ch2_training_images/img_%d.jpg" % i)
#     text, boxes = rec.recognize_image(img)
#     #print(boxes)
#     rec.draw(img, text, boxes)