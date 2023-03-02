# TextPlacement
cDCGAN for generation realistic images with given text.
We need to generate images satisfying certain criteria so that they will not become blurry or unreadable. Basic learning process of a GAN minimizes loss function automatically. Thus, we put those criteria as part of the Discriminator of our GAN, making it assign low scores to blurry and other unwanted images.

Described in the paper [Synthetic dataset generation for text recognition with generative adversarial networks](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11433/1143315/Synthetic-dataset-generation-for-text-recognition-with-generative-adversarial-networks/10.1117/12.2558271.short?SSO=1).

In order to place text on the image, you need to follow a few simple steps:
* Get pre-trained [E2E-MLT model](https://github.com/MichalBusta/E2E-MLT).
* Train the model using [Kaist dataset](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database).
* Call 
    ```bash
    draw_text(image, 'text', position, RGBcolor, fontsize, width_wrap):
    ```
