import os
import img_data_generator
import json
import random


class ImgFileNaming:
    def _change_file_name(self, dir):
        file_list = [
            f for f in os.listdir(dir) if f.endswith(".json") and "bg_" not in f
        ]
        for file in file_list:
            label_path = os.path.join(dir, file)
            img_path = os.path.join(dir, file.replace(".json", ".jpg"))
            with open(label_path, "r+") as f:
                data = json.load(f)
                if data["shapes"] == []:
                    new_label_path = os.path.join(dir, "bg_" + file)
                    new_img_path = os.path.join(
                        dir, "bg_" + file.replace(".json", ".jpg")
                    )
                    data["imagePath"] = "bg_" + file.replace(".json", ".jpg")
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f)
                    f.close()
                    os.rename(label_path, new_label_path)
                    os.rename(img_path, new_img_path)

    def _split_data(self, dir, split_size=0.8):
        img_list = [f for f in os.listdir(dir) if f.endswith(".jpg")]
        par_dir = os.path.dirname(dir)
        if img_list == []:
            pass
        else:
            random.shuffle(img_list)  # me-random list secara inplace
            n_data = len(img_list)
            split = int(n_data * split_size)
            train_data, test_data = img_list[:split], img_list[split:]
            for img_train in train_data:
                img_path = os.path.join(dir, img_train)
                label_path = os.path.join(dir, img_train.replace(".jpg", ".json"))
                if os.path.exists(img_path):
                    new_img_path = os.path.join(par_dir, "train", img_train)
                    os.replace(img_path, new_img_path)
                if os.path.exists(label_path):
                    new_label_path = os.path.join(
                        par_dir, "train", img_train.replace(".jpg", ".json")
                    )
                    os.replace(label_path, new_label_path)
            for img_test in test_data:
                img_path = os.path.join(dir, img_test)
                label_path = os.path.join(dir, img_test.replace(".jpg", ".json"))
                if os.path.exists(img_path):
                    new_img_path = os.path.join(par_dir, "test", img_test)
                    os.replace(img_path, new_img_path)
                if os.path.exists(label_path):
                    new_label_path = os.path.join(
                        par_dir, "test", img_test.replace(".jpg", ".json")
                    )
                    os.replace(label_path, new_label_path)


if __name__ == "__main__":
    _instance_class = ImgFileNaming()
    _instance_img_gen = img_data_generator.ImgGenerator()
    _instance_class._change_file_name(_instance_img_gen.get_directory(0))
    _instance_class._split_data(_instance_img_gen.get_directory(0))
