import math
import os
import subprocess

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def combine_images(generated_images):
    num_images = generated_images.shape[0]
    new_width = int(math.sqrt(num_images))
    new_height = int(math.ceil(float(num_images) / new_width))
    grid_shape = generated_images.shape[1:3]
    grid_image = np.zeros((new_height * grid_shape[0], new_width * grid_shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / new_width)
        j = index % new_width
        grid_image[i * grid_shape[0]:(i + 1) * grid_shape[0], j * grid_shape[1]:(j + 1) * grid_shape[1]] = \
            img[:, :, 0]
    return grid_image


def generate_noise(shape: tuple):
    noise = np.random.uniform(0, 1, size=shape)
    return noise


def generate_condition_embedding(label: int, nb_of_label_embeddings: int):
    label_embeddings = np.zeros((nb_of_label_embeddings, 100))
    label_embeddings[:, label] = 1
    return label_embeddings


def generate_images(generator, nb_images: int, label: int):
    noise = generate_noise((nb_images, 100))
    label_batch = generate_condition_embedding(label, nb_images)
    generated_images = generator.predict([noise, label_batch], verbose=0)
    return generated_images


def generate_mnist_image_grid(generator, title: str = "Generated images"):
    generated_images = []

    for i in range(10):
        noise = generate_noise((10, 100))
        label_input = generate_condition_embedding(i, 10)
        gen_images = generator.predict([noise, label_input], verbose=0)
        generated_images.extend(gen_images)

    generated_images = np.array(generated_images)
    image_grid = combine_images(generated_images)
    image_grid = inverse_transform_images(image_grid)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(image_grid, cmap="gray")
    ax.set_title(title)
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return image


def save_generated_image(image, epoch, iteration, folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    file_path = "{0}/{1}_{2}.png".format(folder_path, epoch, iteration)
    cv2.imwrite(file_path, image.astype(np.uint8))


def transform_images(images: np.ndarray):
    """
    Transform images to [-1, 1]
    images shape (60000, 28, 28)
    needed 640x480 image
    """
#     w = 306
#     h = 226
#     images = np.pad(images, ((0, 0), (h, h), (w, w)), 'minimum')

    images = (images.astype(np.float32) - 127.5) / 127.5
    return images


def inverse_transform_images(images: np.ndarray):
    """
    From the [-1, 1] range transform the images back to [0, 255]
    """

    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images



def preprocess(img, size):
    """Precprocess an image for ImageNet models.
    
    # Arguments
        img: Input Image
        size: Target image size (height, width).
    
    # Return
        Resized and mean subtracted BGR image, if input was also BGR.
    """
    h, w = size
    img = np.copy(img)
    img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    mean = np.array([104,117,123])
    img -= mean[np.newaxis, np.newaxis, :]
    return img

def sample_batch(batch_size, batch_index, input_size=(512,512), preserve_aspect_ratio=False):
    h, w = input_size
    aspect_ratio = w/h
    idxs = np.arange(min(batch_size*batch_index, self.num_samples), 
                     min(batch_size*(batch_index+1), self.num_samples))

    if len(idxs) == 0:
        print('WARNING: empty batch')

    inputs = []
    data = []
    for i in idxs:
        img_path = os.path.join(self.image_path, self.image_names[i])
        img = cv2.imread(img_path)
        if preserve_aspect_ratio:
            img = pad_image(img, aspect_ratio)
        inputs.append(preprocess(img, input_size))
        data.append(self.data[i])
    inputs = np.asarray(inputs)
        
    return inputs, data