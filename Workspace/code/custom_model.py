import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    Dense,
    GlobalMaxPool2D,
    Reshape,
    Activation,
    Concatenate,
)
from keras.applications import VGG16
import numpy as np


# tf.compat.v1.disable_eager_execution()


class AnchorGenerator(layers.Layer):
    def __init__(self, scales, aspect_ratio, **kwargs):
        super(AnchorGenerator, self).__init__(**kwargs)
        self.scales = scales
        self.aspect_ratio = aspect_ratio

    def build(self, input_shape):
        self.num_anchor = len(self.aspect_ratio)

    # def anchor_gen(self, inputs):
    #     """AnchorGenerator call

    #     Args:
    #         inputs (Tensor): tensor with shape of (batch, input_height, input_width, num_channel)

    #     Returns:
    #         list : return a list or tensor with properties of [cx, cy, width, height, scales] shaping (batch, num_anchor, properties)
    #     """
    #     batch_size = tf.shape(inputs)[0]
    #     img_h = inputs.shape[1]
    #     img_w = inputs.shape[2]
    #     anchor_boxes = []
    #     for aspect_ratio in self.aspect_ratio:
    #         box_width = self.scales * tf.sqrt(aspect_ratio)
    #         box_height = self.scales / tf.sqrt(aspect_ratio)
    #         for i in range(img_w):
    #             x = (i + 0.5) / float(img_w)
    #             for j in range(img_h):
    #                 y = (j + 0.5) / float(img_h)
    #                 anchor_boxes.append(
    #                     tf.concat([[x, y, box_width, box_height, self.scales]], axis=0)
    #                 )
    #     anchor_boxes = tf.convert_to_tensor(anchor_boxes, dtype=tf.float32)
    #     num_loc = img_h * img_w
    #     anchor_boxes = tf.reshape(anchor_boxes, [1, self.num_anchor * num_loc, 5])
    #     anchor_boxes = tf.tile(anchor_boxes, (batch_size, 1, 1))
    #     anchor_boxes = tf.cast(anchor_boxes, dtype=tf.float32)
    #     return anchor_boxes
    
    def anchor_gen1(self, input_tensor):
        """AnchorGenerator call

        Args:
            inputs (Tensor): tensor with shape of (batch, input_height, input_width, num_channel)

        Returns:
            list : return a list or tensor with properties of [cx, cy, width, height, scales] shaping (batch, num_anchor, properties)
        """
        batch_size = tf.shape(input_tensor)[0]
        img_h = input_tensor.shape[1]
        img_w = input_tensor.shape[2]
        
        num_aspect = len(self.aspect_ratio)
        num_loc = img_h * img_w
        box_widths = self.scales * tf.sqrt(self.aspect_ratio)
        box_heights = self.scales / tf.sqrt(self.aspect_ratio)
        print(box_widths)
        
        x_coords = tf.range(0, img_w, dtype=tf.float32) + 0.5
        y_coords = tf.range(0, img_h, dtype=tf.float32) + 0.5
        x_coords = x_coords / float(img_w)
        y_coords = y_coords / float(img_h)
        x_coords, y_coords = tf.meshgrid(x_coords, y_coords)
        # print(x_coords, y_coords)
        x_coords = tf.reshape(x_coords, [-1])
        y_coords = tf.reshape(y_coords, [-1])
        # print(x_coords, y_coords)
        
        
        cx = tf.tile(tf.expand_dims(x_coords, 1), [num_aspect, 1])
        cy = tf.tile(tf.expand_dims(y_coords, 1), [num_aspect, 1])
        # print(cx, cy)
        
        cx = tf.reshape(cx, [-1])
        cy = tf.reshape(cy, [-1])
        
        box_widths = tf.reshape(box_widths, [num_aspect, 1])
        box_widths = tf.tile(box_widths, [1, num_loc])
        box_widths = tf.reshape(box_widths, [num_loc*num_aspect])
        box_heights = tf.reshape(box_heights, [num_aspect, 1])
        box_heights = tf.tile(box_heights, [1, num_loc])
        box_heights = tf.reshape(box_heights, [num_loc*num_aspect])
        
        scales = tf.tile([self.scales], [num_loc*num_aspect])
        anchor_boxes = tf.stack([cx, cy, box_widths, box_heights, scales], axis=-1)
        anchor_boxes = tf.expand_dims(anchor_boxes, 0)
        anchor_boxes = tf.tile(anchor_boxes, (batch_size, 1, 1))
        # cx = tf.expand_dims(x_coords, 1)
        # anchor_boxes =
        return anchor_boxes





    def call(self, inputs, img_size, **kwargs):
        # arr = self.anchor_gen(inputs)
        arr = self.anchor_gen1(inputs)
        return arr




class CustomModel(tf.keras.Model):
    def __init__(self, input_shape=None, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        # self.input_layer = tf.keras.layers.Input((300, 300, 3))
        # if tf.math.not_equal(input_shape, None):
        #     self.input_layer = tf.keras.layers.Input(input_shape)
        # else:
        #     self.input_layer = None
        # TODO CALL PREPROCESSING FOR VGG

        vgg = VGG16(include_top=False, input_shape=input_shape)
        self.vggs = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv3").output)  # type: ignore
        self.vgg3 = Model(inputs=vgg.input, outputs=vgg.get_layer("block4_conv3").output)  # type: ignore

        self.conv6_1 = Conv2D(1024, (3, 3), padding="same", name="block6_conv1")

        self.conv7_1 = Conv2D(1024, (1, 1), padding="same", name="block7_conv1")

        self.conv8_1 = Conv2D(256, (1, 1), padding="same", name="block8_conv1")
        self.conv8_2 = Conv2D(
            512, (3, 3), strides=(2, 2), padding="same", name="block8_conv2"
        )

        self.conv9_1 = Conv2D(128, (1, 1), padding="same", name="block9_conv1")
        self.conv9_2 = Conv2D(
            256, (3, 3), strides=(2, 2), padding="same", name="block9_conv2"
        )

        self.conv10_1 = Conv2D(128, (1, 1), padding="same", name="block10_conv1")
        self.conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), name="block10_conv2")

        self.conv11_1 = Conv2D(128, (1, 1), padding="same", name="block11_conv1")
        self.conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), name="block11_conv2")

        # self.cl4_3 = Conv2D(4 * 1, (3, 3), padding="same", name='block4_cl4_3')
        # self.anchor_gen4_3 = AnchorGenerator(scales=0.1, aspect_ratio=[1.0, 2.0, 3.0, 0.5, 1/3, 1.5], label=self.batch[1])
        # self.classifier_4 = Conv2D(6*4, (3, 3), strides=(1, 1), padding='same')
        self.anchor_gen4_3 = AnchorGenerator(
            scales=0.1, aspect_ratio=[1.0, 2.0]
        )
        self.classifier_4 = Conv2D(2 * 6, (3, 3), strides=(1, 1), padding="same")
        self.reshape_c_4 = Reshape((-1, 6))
        self.activation_c_4 = Activation("sigmoid")

        self.rl4_3 = Conv2D(4 * 4, (3, 3), padding="same", name="block4_rl4_3")

        self.cl7_1 = Conv2D(4 * 1, (3, 3), padding="same", name="block_cl7_1")
        self.rl7_1 = Conv2D(4 * 4, (3, 3), padding="same", name="block_rl7_1")

        self.reshape_1_1 = Reshape((-1, 1), name="reshape_1_1")
        self.reshape_1_2 = Reshape((-1, 1), name="reshape_1_2")

        self.concat = Concatenate(axis=1)
        print("input ==== ", vgg.input)
        self.out = self.call(vgg.input)

    def call(self, input_tensor, **kwargs):
        vgg_output = self.vggs(input_tensor)
        conv4_3_output = self.vgg3(input_tensor)
        conv6_1_output = self.conv6_1(vgg_output)
        conv7_1_output = self.conv7_1(conv6_1_output)
        conv8_1_output = self.conv8_1(conv7_1_output)
        conv8_2_output = self.conv8_2(conv8_1_output)
        conv9_1_output = self.conv9_1(conv8_2_output)
        conv9_2_output = self.conv9_2(conv9_1_output)
        conv10_1_output = self.conv10_1(conv9_2_output)
        conv10_2_output = self.conv10_2(conv10_1_output)
        conv11_1_output = self.conv11_1(conv10_2_output)
        conv11_2_output = self.conv11_2(conv11_1_output)

        cl4_3_output = self.anchor_gen4_3(conv10_1_output, input_tensor.shape[1])  # type: ignore
        # assert cl4_3_output is not None
        # cl4_3_output = Reshape((-1, 1))(cl4_3_output)
        # cl4_3_output = Activation("sigmoid")(cl4_3_output)

        # rl4_3_output = self.rl4_3(vgg_3)
        # rl4_3_output = Reshape((-1, 4))(rl4_3_output)
        # rl4_3_output = Activation("sigmoid")(rl4_3_output)
        # assert rl4_3_output is not None

        # cl7_1_output = self.cl7_1(conv7_1_output)
        # cl7_1_output = Reshape((-1, 1))(cl7_1_output)
        # cl7_1_output = Activation("sigmoid")(cl7_1_output)

        # rl7_1_output = self.rl7_1(conv7_1_output)
        # rl7_1_output = Reshape((-1, 4))(rl7_1_output)
        # rl7_1_output = Activation("sigmoid")(rl7_1_output)

        # # c_layer = self.concat([cl4_3_output, cl7_1_output])
        # r_layer = self.concat([rl4_3_output, rl7_1_output])

        # print(rl4_3_output.shape)
        # print('--out--')
        # print(c_layer.shape)
        # print(r_layer.shape)

        # print(model.summary())

        pos_arr = self.classifier_4(conv10_1_output)  # type: ignore
        pos_arr = self.reshape_c_4(pos_arr)
        pos_arr = self.activation_c_4(pos_arr)
        
        
        return [pos_arr, cl4_3_output]

    # TODO CHANGE BACK LATER
    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs) # type: ignore
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            pred, anch = self.call(X, training=True)
            anch_arr = self.anchor2array(anch, X.shape[1])
            
            print(pred[0])
            print(anch[0])
            # print(pred.shape)
            print('train->>>')
            # print(anch[0])
            # print(anch_arr[0])
            # print(anch_arr1[0])
            
            box = self.box2array(y, X.shape[1])
            print('booox ==',box)
            aa
            # print(anch_arr)
            iou_arr = self.compute_iou(anch_arr, box)
            print(iou_arr[0])
            pos_anch_arr, pos_pred_arr  = self.pos_neg_boxes(iou_arr, pred, anch_arr, 0.05)
            print(pos_anch_arr[0])
            print(pos_pred_arr[0])
            loss = self.lloss(pos_anch_arr, pred)

            total_loss = loss
            print("lossss ==>", loss)

            grad = tape.gradient(total_loss, self.trainable_variables)

        self.opt.apply_gradients(zip(grad, self.trainable_variables))

        return {
            "total_loss": total_loss,
            # "class_loss": batch_classloss,
            # "regress_loss": batch_localizationloss,
        }

    def test_step(self, batch, **kwargs):
        X, y = batch

        pred, anch = self.model(X, training=False)
        box = self.box2array(y[1], X.shape[1])
        iou_arr = self.compute_iou(anch, box)
        pos_anch_arr, test  = self.pos_neg_boxes(iou_arr, anch, 0.12)
        
        loss = self.lloss(pos_anch_arr, pred)
        
        # batch_classloss = self.closs(y[0], classes)
        # batch_localizationloss = self.lloss(y[1], coords)
        total_loss = loss

        return {
            "total_loss": total_loss,
            # "class_loss": batch_classloss,
            # "regress_loss": batch_localizationloss,
        }

    def predict(
        self,
        x,
        y,
        batch_size=None,
        verbose="auto",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):  
        pred, anch = self.call(x)
        
        # print('label -======>>', y)
        box = self.box2array(y[1], x.shape[1])
        iou_arr = self.compute_iou(pred, box)
        pos_anch_arr, test  = self.pos_neg_boxes(iou_arr, pred, 0.12)
        
        loss = self.lloss(pos_anch_arr, pred)
        return pred, pos_anch_arr.numpy()
    
    def box2array(self, label, img_size):
        # print("box ==>", label)
        box = label[:, 1:5]
        box = box * img_size
        xmin, xmax, ymin, ymax = box[:, 0], box[:, 2], box[:, 1], box[:, 3]
        xmin = tf.reshape(xmin, [box.shape[0], 1])
        xmax = tf.reshape(xmax, [box.shape[0], 1])
        ymin = tf.reshape(ymin, [box.shape[0], 1])
        ymax = tf.reshape(ymax, [box.shape[0], 1])

        # arr = tf.convert_to_tensor([xmin, xmax, ymin, ymax])
        arr = tf.concat([label[:, :1], xmin, xmax, ymin, ymax, label[:, 5:]], axis=-1)
        # return xmin
        return arr
    
    def prop2abs(self, x, y, box_w, box_h, img_size):
        width = box_w * img_size / 2
        height = box_h * img_size / 2
        cx = x * img_size
        cy = y * img_size
        return (cx - width), (cx + width), (cy - height), (cy + height)

    def abs2prop(self, xmin, xmax, ymin, ymax, img_size):
        width = float(xmax-xmin)
        height = float(ymax-ymin)
        cx = float(xmin) + width/2
        cy = float(ymin) + height/2
        
        width /= img_size
        height /= img_size
        cx /= img_size
        cy /= img_size
        return cx, cy, width, height
    # @tf.function
    def anchor2array(self, inputs_anchors, img_size):
        # batch_size = tf.shape(inputs_anchors)[0]
        # if batch_size is None:
        #     batch_size = 1
        num_anchors = tf.shape(inputs_anchors)[1]
        # print()
        # print(inputs_anchors)
        # input_anchors = tf.expand_dims(inputs_anchors, axis=-1)
        # print(inputs_anchors)
        # input_anchors = tf.tile(input_anchors, [1, 1, 1, 4])
        x = inputs_anchors[:, :, 0]
        y = inputs_anchors[:, :, 1]
        box_w = inputs_anchors[:, :, 2]
        box_h = inputs_anchors[:, :, 3]
        xmin, xmax, ymin, ymax = self.prop2abs(
            x=x, y=y, box_w=box_w, box_h=box_h, img_size=img_size
        )

        xmin = tf.reshape(xmin, [-1, num_anchors, 1])
        xmax = tf.reshape(xmax, [-1, num_anchors, 1])
        ymin = tf.reshape(ymin, [-1, num_anchors, 1])
        ymax = tf.reshape(ymax, [-1, num_anchors, 1])

        # Concatenate xmin, xmax, ymin, ymax along the last dimension
        arr = tf.concat([xmin, xmax, ymin, ymax], axis=-1)
        return arr

    # @tf.function
    def compute_iou(self, anchor_arr, box):
        box = tf.reshape(box, [-1, 1, 6])
        # print('aaa=',box[:, :, 1])
        # print(anchor_arr[:, :, 0])
        xmin = tf.math.maximum(box[:, :, 1], anchor_arr[:, :, 0])
        # print(xmin)
        xmax = tf.math.minimum(box[:, :, 2], anchor_arr[:, :, 1])
        ymin = tf.math.maximum(box[:, :, 3], anchor_arr[:, :, 2])
        ymax = tf.math.minimum(box[:, :, 4], anchor_arr[:, :, 3])

        area_anch = (anchor_arr[:, :, 1] - anchor_arr[:, :, 0] + 1) * (
            anchor_arr[:, :, 3] - anchor_arr[:, :, 2] + 1
        )
        area_box = (box[:, :, 2] - box[:, :, 1] + 1) * (box[:, :, 4] - box[:, :, 3] + 1)
        # print('aaa=',area_box)
        w = tf.math.maximum(0.0, xmax - xmin + 1)
        h = tf.math.maximum(0.0, ymax - ymin + 1)
        intersection = w * h
        union = area_box + area_anch - intersection

        return intersection / union
    
    def adjust_gt_box(self, anchor_arr, box_arr):
        adjusted_anchor = anchor_arr[:, :, 1] 
        

    # @tf.function
    def pos_neg_boxes(self, iou_arr, pred_arr, anchor_arr, threshold):
        iou = iou_arr > threshold
        # print("anchor arr ======> ", anchor_arr)
        # new_tensor = tf.constant([True, True, True, True], dtype=tf.bool)
        # iou = tf.concat([iou, new_tensor], axis=0)
        iou_mask = tf.expand_dims(iou, axis=-1)
        # print(tf.expand_dims(iou, axis=-1))
        # pos_mask = tf.cast(iou, tf.bool)
        # iou = tf.where(pos_mask, anchor_arr, tf.zeros_like(anchor_arr))
        pos_anch_arr = tf.where(iou_mask, anchor_arr, tf.zeros_like(anchor_arr))
        pos_pred_arr = tf.where(iou_mask, pred_arr, tf.zeros_like(pred_arr))

        # iou = tf.math.argmax(iou_arr, axis=1)
        return pos_anch_arr, pos_pred_arr

    # def summary_model(self):
    #     inputs = keras.Input(shape=(300, 300, 3))
    #     outputs = self.call(inputs)
    #     CustomModel(inputs=inputs, outputs=outputs, name='Custom Model SSD').summary()
