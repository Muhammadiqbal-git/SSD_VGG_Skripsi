import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import img_data_generator
import os


class ImgLoad:
    def _limit_gpu(self):
        """Limit GPU memory growth for tensorflow"""
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(gpus)

    def decode_img(self, img):
        """Decode an img from the parameter

        Args:
            img (String): Path filename of imgs

        Returns:
            Tensor: Return tensor of imgs
        """
        byte_img = tf.io.read_file(img)
        image = tf.io.decode_jpeg(byte_img)
        return image

    def _load_imgs(
        self,
    ):
        """Load and process imgs

        Returns:
            tf.data.Dataset: A mapped of tf dataset containing the imgs
        """
        img_gen_instance = img_data_generator.ImgGenerator()
        file_name = os.path.join(img_gen_instance.get_directory(0), "*.jpg")
        images = tf.data.Dataset.list_files(file_name, shuffle=False)
        images = images.map(self.decode_img) # type: ignore
        return images

    def _img_preview(self, images):
        image_generator = images.batch(4).as_numpy_iterator()
        plot_images = image_generator.next()
        fix, ax = plt.subplots(ncols=4, figsize=(20, 20))
        for idx, image in enumerate(plot_images):
            ax[idx].imshow(image)
        plt.show()


if __name__ == "__main__":
    _main_instance = ImgLoad()
    _main_instance._limit_gpu()
    data = _main_instance._load_imgs()
    _main_instance._img_preview(data)
