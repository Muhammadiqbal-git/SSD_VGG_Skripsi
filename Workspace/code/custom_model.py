import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, GlobalMaxPool2D, Reshape, Activation, Concatenate
from keras.applications import VGG16
import numpy as np

class AnchorGenerator(layers.Layer):
    def __init__(self, scales, aspect_ratio, **kwargs):
        super(AnchorGenerator, self).__init__(**kwargs)
        self.scales = scales
        self.aspect_ratio = aspect_ratio
        
    def build(self, input_shape):
        self.num_anchor = len(self.aspect_ratio)
    
    def call(self, inputs):
        """ AnchorGenerator call

        Args:
            inputs (Tensor): tensor

        Returns:
            list : return a list or tensor
        """
        batch_size = tf.shape(inputs)[0]
        img_h = inputs.shape[1]
        img_w = inputs.shape[2]     
        anchor_boxes = []
        for aspect_ratio in self.aspect_ratio:
            box_width = self.scales * tf.sqrt(aspect_ratio)
            box_height = self.scales / tf.sqrt(aspect_ratio)
            for i in range(img_h):
                x = (i + 0.5) / float(img_h) 
                for j in range(img_w):
                    y = (j + 0.5) / float(img_w)
                    anchor_boxes.append([x, y, box_width, box_height, self.scales])
        anchor_boxes = tf.convert_to_tensor(anchor_boxes, dtype=tf.float32)
        print('tes')
        print(anchor_boxes)
        num_loc = img_h * img_w
        anchor_boxes = tf.reshape(anchor_boxes, [1, self.num_anchor * num_loc, 5])
        print(anchor_boxes)
        anchor_boxes = tf.tile(anchor_boxes, (batch_size, 1, 1))
        anchor_boxes = tf.cast(anchor_boxes, dtype=tf.float32)
        return anchor_boxes
        
            


class CustomModel(Model):
    def __init__(self, input_shape=None,**kwargs):
        super(CustomModel, self).__init__(**kwargs)
        # self.input_layer = tf.keras.layers.Input((300, 300, 3))
        if input_shape is not None:
            self.input_layer = tf.keras.layers.Input(input_shape)
        else:
            self.input_layer = None
        # TODO CALL PREPROCESSING FOR VGG
        vgg = VGG16(include_top=False, input_shape=input_shape)
        self.vggs = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv3").output) # type: ignore
        self.vgg3 = Model(inputs=vgg.input, outputs=vgg.get_layer("block4_conv3").output) # type: ignore
        

        self.conv6_1 = Conv2D(1024, (3, 3), padding="same", name="block6_conv1")

        self.conv7_1 = Conv2D(1024, (1, 1), padding="same", name="block7_conv1")

        self.conv8_1 = Conv2D(256, (1, 1), padding="same", name="block8_conv1")
        self.conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="block8_conv2")

        self.conv9_1 = Conv2D(128, (1, 1), padding="same", name="block9_conv1")
        self.conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", name="block9_conv2")

        self.conv10_1 = Conv2D(128, (1, 1), padding="same", name="block10_conv1")
        self.conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), name="block10_conv2")

        self.conv11_1 = Conv2D(128, (1, 1), padding="same", name="block11_conv1")
        self.conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), name="block11_conv2")

        # self.cl4_3 = Conv2D(4 * 1, (3, 3), padding="same", name='block4_cl4_3')
        self.anchor_gen4_3 = AnchorGenerator(scales=0.1, aspect_ratio=[1.0, 2.0, 0.5])
        self.rl4_3 = Conv2D(4 * 4, (3, 3), padding="same", name='block4_rl4_3')

        self.cl7_1 = Conv2D(4 * 1, (3, 3), padding='same', name='block_cl7_1')
        self.rl7_1 = Conv2D(4 * 4, (3, 3), padding='same', name='block_rl7_1')
        
        self.reshape_1_1 = Reshape((-1, 1), name='reshape_1_1')
        self.reshape_1_2 = Reshape((-1, 1), name='reshape_1_2')
        
        self.concat = Concatenate(axis=1)
        self.out = self.call(self.input_layer)
    


    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs) # type: ignore
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def prop2abs(self, x, y, box_w, box_h, img_size):
        width = (box_w*img_size/2)
        height = (box_h*img_size/2)
        cx = (x * img_size)
        cy = (y * img_size)
        return (cx-width), (cx+width), (cy-height), (cy+height)
    
    def anchor2array(self, inputs_anchors, img_size):
        """_summary_

        Args:
            anchors (List): [batch, num_anchors, [x, y, box_w, box_h, scale]]
            img_size (_type_): _description_
        """
        arr = tf.zeros((tf.shape(inputs_anchors)[0], tf.shape(inputs_anchors)[1], 4))
        for i in range(tf.shape(inputs_anchors)[0]):
            for j in range(tf.shape(inputs_anchors)[1]):
                anchor = inputs_anchors[i, j]
                xmin, xmax, ymin, ymax = self.prop2abs(x=anchor[0], y=anchor[1], box_w=anchor[2], box_h=anchor[3], img_size=img_size)
                arr = tf.tensor_scatter_nd_update(arr, [[i, j]], [[xmin, xmax, ymin, ymax]])
        return arr
    
    def box2array(self, label, img_size):
        box = label[:, 1] * img_size
        xmin, xmax, ymin, ymax = box[0], box[2], box[1], box[3]
        arr = tf.convert_to_tensor([xmin, xmax, ymin, ymax])
        return arr
    
    def compute_iou(self, anchor_arr, box_arr):
        for box in box_arr:
            xmin = tf.math.maximum(box_arr[0], anchor_arr[:, 0])
            xmax = tf.math.minimum(box_arr[1], anchor_arr[:, 1])
            ymin = tf.math.maximum(box_arr[2], anchor_arr[:, 2])
            ymax = tf.math.minimum(box_arr[3], anchor_arr[:, 3])
            
            area_anch = (anchor_arr[:,1] - anchor_arr[:, 0]+1) * (anchor_arr[:, 3] - anchor_arr[:, 2]+1)
            area_box = (box_arr[1] - box_arr[0]+1) * (box_arr[3] - box_arr[2]+1)
            
        
    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(y[1], coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {
            "total_loss": total_loss,
            "class_loss": batch_classloss,
            "regress_loss": batch_localizationloss,
        }

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(y[1], coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {
            "total_loss": total_loss,
            "class_loss": batch_classloss,
            "regress_loss": batch_localizationloss,
        }

    def call(self, input_tensor, **kwargs):
        vgg_output = self.vggs(input_tensor)
        vgg_3 = self.vgg3(input_tensor)
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

        cl4_3_output = self.anchor_gen4_3(vgg_3)
        assert cl4_3_output is not None
        print(cl4_3_output.shape)
        # cl4_3_output = Reshape((-1, 1))(cl4_3_output)
        # cl4_3_output = Activation("sigmoid")(cl4_3_output)

        rl4_3_output = self.rl4_3(vgg_3)
        rl4_3_output = Reshape((-1, 4))(rl4_3_output)
        rl4_3_output = Activation("sigmoid")(rl4_3_output)
        assert rl4_3_output is not None

        cl7_1_output = self.cl7_1(conv7_1_output)
        cl7_1_output = Reshape((-1, 1))(cl7_1_output)
        cl7_1_output = Activation('sigmoid')(cl7_1_output)

        rl7_1_output = self.rl7_1(conv7_1_output)
        rl7_1_output = Reshape((-1, 4))(rl7_1_output)
        rl7_1_output = Activation('sigmoid')(rl7_1_output)

        # c_layer = self.concat([cl4_3_output, cl7_1_output])
        r_layer = self.concat([rl4_3_output, rl7_1_output])
        print('-==========-')
        anch = tf.fill([8, 40, 5], 2.0)
        test = self.anchor2array(inputs_anchors=anch, img_size=input_tensor.shape[1])
        print(test)
        # # print(cl4_3_output.shape)
        # print(cl7_1_output.shape)
        print(rl4_3_output.shape)
        # print('--out--')
        # print(c_layer.shape)
        # print(r_layer.shape)
        
        
        
        # model = Model(inputs=input_tensor, outputs=[c_layer, r_layer])
        # print(model.summary())
        return [cl4_3_output, r_layer]
    
    # def summary_model(self):
    #     inputs = keras.Input(shape=(300, 300, 3))
    #     outputs = self.call(inputs)
    #     CustomModel(inputs=inputs, outputs=outputs, name='Custom Model SSD').summary()
    

