import os
import time
import random
import cv2
import time


class ImgGenerator:
    number_images = 25
    PARENT_DIR = os.path.dirname(os.getcwd())
    DIR_NAME = 'imgs'
    LIST_SUB_DIR = ["all_img", "train", "test", 'aug_train', 'aug_test']

    def _create_directory(self):
        path = os.path.join(self.PARENT_DIR, self.DIR_NAME)
        if not os.path.isdir(path):
            os.makedirs(path)
            for item in self.LIST_SUB_DIR:
                sub_path = os.path.join(path, item)
                print(sub_path)
                os.makedirs(sub_path)

    def get_directory(self, option):
        """Get a working directory path based on the specified option

        Args:
            option (int): The option to determine the subfolder directory.
                      0 = all_img, 1 = train, 2 = test, 3 = aug_train, 4 = aug_test

        Returns:
            str: The directory path as a string
        Raises:
            IndexError : If an invalid option is provided
        """
        return os.path.join(self.PARENT_DIR, self.DIR_NAME, self.LIST_SUB_DIR[option]) # type: ignore

    def _start_capture(self):
        cap = cv2.VideoCapture(0)
        # while True:
        for img_num in range(self.number_images):
            
            ret, frame = cap.read()
            path = self.get_directory(0)
            time_now = time.strftime("%Y-%m-%d")
            img_name = os.path.join(
                path, "{}_{}.jpg".format(time_now, random.randint(1111, 9999))
            )
            # if cv2.waitKey(1) & 0xFF == ord('c'):
            #     cv2.imwrite(img_name, frame)
            
            cv2.imshow("frame", frame)
            if img_num == 0:
                time.sleep(3.0)
            cv2.imwrite(img_name, frame)
            time.sleep(0.3)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _main_instance = ImgGenerator()
    _main_instance._create_directory()
    _main_instance._start_capture()

IMAGES_PATH = os.path
