import tensorflow as tf
import os
import img_load
import img_data_generator
import json
from matplotlib import pyplot as plt
import cv2
import numpy as np


class DataLoad:
    def set_tf_img(self, dir, dim: tuple):
        path = os.path.join(dir, "*.jpg")
        data = tf.data.Dataset.list_files(path, shuffle=False)
        data = data.map(lambda x: img_load.ImgLoad.decode_img(img_load.ImgLoad, x))  # type: ignore
        data = data.map(lambda x: tf.image.resize(x, dim))
        data = data.map(lambda x: x / 255)
        return data

    def load_labels(self, label_path):
        with open(label_path.numpy(), "r", encoding="utf-8") as f:
            label = json.load(f)
            return [label["class"]], label["bbox"]

    def set_tf_label(self, dir):
        path = os.path.join(dir, "*.json")
        labels = tf.data.Dataset.list_files(path, shuffle=False)
        labels = labels.map(lambda x: tf.py_function(func=self.load_labels, inp=[x], Tout=[tf.uint8, tf.float32]))  # type: ignore
        return labels

    def set_tf_data(self, img_data, label_data,  batch=8, prefetch=0, shuffle=-1,):
        if shuffle == -1:
            shuffle = tf.data.Dataset.cardinality(img_data)
        dataset = tf.data.Dataset.zip((img_data, label_data))
        dataset = dataset.shuffle(shuffle)
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(prefetch)
        return dataset

    def preview_data(self, data: tf.data.Dataset):
        review = data.as_numpy_iterator().next()
        fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
        print('=s=s=s=s=')
        print(review[1])
        print(review[1][1][0][0])
        # print(review[0][1])
        # print(review[1][1][1])

        for idx in range(4):
            sample_image = review[0][idx]
            sample_coors = review[1][1][idx]
            cv2.rectangle(
                # (sample_image * 120).astype(np.uint8),
                sample_image,
                tuple(np.multiply(sample_coors[:2], [300, 300]).astype(int)),
                tuple(np.multiply(sample_coors[2:], [300, 300]).astype(int)),
                (255, 0, 0),
                1,
            )
           
            ax[idx].imshow((sample_image * 255).astype(np.uint8))
            # ax[idx].imshow(sample_image)
            
        plt.show()
        
    def load(self):
        _instance_img_gen = img_data_generator.ImgGenerator()
        
        train_path = _instance_img_gen.get_directory(3)
        test_path = _instance_img_gen.get_directory(4)

        train_img = self.set_tf_img(train_path, (300, 300))
        test_img = self.set_tf_img(test_path, (300, 300))
        train_label = self.set_tf_label(train_path)
        test_label = self.set_tf_label(test_path)
        # print(len(train_img), len(train_label), len(test_img), len(test_label))  # type: ignore

        train = self.set_tf_data(train_img, train_label, 8, 4)
        test = self.set_tf_data(test_img, test_label, 8, 4)
        print('train')
        print(tf.data.Dataset.cardinality(train))
        return train, test
        
        


if __name__ == "__main__":
    _instance_class = DataLoad()
    _instance_img_load = img_load.ImgLoad()
    _instance_img_load._limit_gpu()
    train, test = _instance_class.load()
    
    
    #preview image
    _instance_class.preview_data(train)
