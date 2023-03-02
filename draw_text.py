from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np

def draw_text(x, text, position, color3, fontsize, width_wrap):

    # text = 'c'
    # position = (10, 10)
    # color = (50, 50, 50)
    # fontsize = 15
    # width_wrap = 20

    data = x #np.zeros((32, 32, 3), dtype=np.uint8)
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

    img_text = np.array(image)

    image.show()
    return img_text

#draw_text('beeee bbbb dgdf rthghjh tjyjhg srghhf rfbgf', position=(400, 100), color=(50, 200, 50), fontsize=100, width_wrap=20,
#          source='../data/ICDAR/end-to-end/ch2_training_images/img_100.jpg')

def draw_text_batch(batch_sz, array, text, position, color3, fontsize, width_wrap):
    return np.array([draw_text(array[i], text[i], position, color3, fontsize, width_wrap) for i in range(0, batch_sz)])

#draw_text(np.zeros((32, 32, 3), dtype=np.uint8), 'a', position=(5, 15), color3=(200, 200, 200), fontsize=18, width_wrap=20)