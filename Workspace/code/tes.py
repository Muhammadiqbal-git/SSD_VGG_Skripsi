from tqdm import tqdm
import tensorflow as tf
import keras.backend as k

inputs = (15, 14, 3)
scales = [0.1, 0.3, 0.5]
aspect_ratios = [0.5, 1.0, 2.0]
tes = [[1, 2, 3],
       [4, 5, 6]]
inputs = tf.keras.Input(shape=inputs)
print(inputs)
num_anchor = len(scales)*len(aspect_ratios)
img_h = tf.shape(inputs)[1]
img_w = tf.shape(inputs)[2]
# cx = tf.range(img_h, dtype=tf.float32)+0.5
# cy = tf.range(img_w, dtype=tf.float32)+0.5
# cx, cy = tf.meshgrid(cx, cy)
# cx = tf.reshape(cx, [-1,]) 
# cy = tf.reshape(cy, [-1, ])       
anchor_boxes = []
for scale in scales:
    for aspect_ratio in aspect_ratios:
        box_width = scale * tf.sqrt(aspect_ratio)
        box_height = scale / tf.sqrt(aspect_ratio)
        anchor_boxes.append([box_width, box_height])
anchor_boxes = tf.convert_to_tensor(anchor_boxes, dtype=tf.float32)
anchor_boxes = tf.reshape(anchor_boxes, [1, 1, num_anchor, 2])

num_loc = img_h * img_w
anchor_boxes = tf.tile(anchor_boxes, (1, num_loc, 1, 1))
anchor_boxes = tf.cast(anchor_boxes, dtype=tf.float32)

print("=============")
# k.eval(anchor_boxes)
k.print_tensor(anchor_boxes)
print(anchor_boxes)