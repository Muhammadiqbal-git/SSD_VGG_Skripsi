import albumentations as alb
import img_data_generator
import os
import cv2
import json
import numpy as np
from tqdm import tqdm


class ImgAugmentator:
    """
    [xmin, ymin, xmax, ymax]
    """
    def img_augment(self, dir):
        augmentator = alb.Compose(
            [
                alb.RandomCrop(height=300, width=300),
                alb.HorizontalFlip(p=0.5),
                alb.RandomBrightnessContrast(p=0.2),
                alb.RandomGamma(p=0.2),
                alb.RGBShift(p=0.4),
            ],
            bbox_params=alb.BboxParams(
                format="albumentations", label_fields=["class_labels"]
            ),
        )
        par_dir = os.path.dirname(dir)
        for sub_dir in ["train", "test"]:
            print('Augmenting .. ', sub_dir)
            path = os.path.join(par_dir, sub_dir)
            aug_par_path = os.path.join (par_dir, "aug_" + sub_dir)
            if not os.path.isdir(aug_par_path):
                os.makedirs(aug_par_path)
            img_list = [f for f in os.listdir(path) if f.endswith(".jpg")]
            for image in tqdm(img_list):
                img_path = os.path.join(path, image)
                label_path = os.path.join(path, image.replace(".jpg", ".json"))
                img = cv2.imread(img_path)
                coor = [0, 0, 0.00001, 0.00001]
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        label = json.load(f)
                        if label["shapes"] != []:
                            coor[0] = label["shapes"][0]["points"][0][0]
                            coor[1] = label["shapes"][0]["points"][0][1]
                            coor[2] = label["shapes"][0]["points"][1][0]
                            coor[3] = label["shapes"][0]["points"][1][1]
                        img_height = label["imageHeight"]
                        img_width = label["imageWidth"]
                        coor = list(
                            np.divide(
                                coor, [img_width, img_height, img_width, img_height]
                            )
                        )
                        try:
                            for x in range(10):
                                augmented = augmentator(
                                    image=img, bboxes=[coor], class_labels=["Human"]
                                )
                                annotation = {}
                                aug_path = os.path.join(
                                    par_dir, "aug_" + sub_dir, "{}_{}".format(x, image)
                                )
                                cv2.imwrite(aug_path, augmented["image"])
                                annotation["image"] = image

                                if os.path.exists(label_path):
                                    if len(augmented["bboxes"]) == 0:
                                        annotation["bbox"] = [0, 0, 0, 0]
                                        annotation["class"] = 0
                                    else:
                                        annotation["bbox"] = augmented["bboxes"][0]
                                        annotation["class"] = 1
                                else:
                                    annotation["bbox"] = [0, 0, 0, 0]
                                    annotation["class"] = 0

                                with open(aug_path.replace('.jpg', '.json'), "w") as f:
                                    json.dump(annotation, f)
                        except Exception as e:
                            print(e)

                # img = cv2.imread


if __name__ == "__main__":
    _instance_class = ImgAugmentator()
    _instance_img_gen = img_data_generator.ImgGenerator()
    _instance_class.img_augment(_instance_img_gen.get_directory(0))
