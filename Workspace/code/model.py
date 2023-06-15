import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, GlobalMaxPool2D, Reshape, Activation
from keras.applications import VGG16
import data_load
import custom_model
import img_load
import numpy


class ModelSkripsi:
    def build_model(self):
        vgg = VGG16(include_top=False, input_shape=(300, 300, 3))
        # conv4_3 = vgg.get_layer("block4_conv3").output # type: ignore
        # vgg = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv3").output)  # type: ignore

        # conv6_1 = Conv2D(1024, (3, 3), padding="same", name="block6_conv1")(vgg.output)

        # conv7_1 = Conv2D(1024, (1, 1), padding="same", name="block7_conv1")(conv6_1)

        # conv8_1 = Conv2D(256, (1, 1), padding="same", name="block8_conv1")(conv7_1)
        # conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="block8_conv2")(conv8_1)

        # conv9_1 = Conv2D(128, (1, 1), padding="same", name="block9_conv1")(conv8_2)
        # conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", name="block9_conv2")(conv9_1)

        # conv10_1 = Conv2D(128, (1, 1), padding="same", name="block10_conv1")(conv9_2)
        # conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name="block10_conv2")(conv10_1)

        # conv11_1 = Conv2D(128, (1, 1), padding="same", name="block11_conv1")(conv10_2)
        # conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name="block11_conv2")(conv11_1)

        # # Classification cl4_3
        # cl4_3 = Conv2D(4 * 1, (3, 3), padding="same", name="block4_cl4_3")(conv4_3)
        # cl4_3 = Reshape((-1, 1))(cl4_3)
        # cl4_3 = Activation("sigmoid")(cl4_3)
        # # Regression cl4_3
        # rl4_3 = Conv2D(4 * 4, (3, 3), padding="same", name="block4_rl4_3")(conv4_3)
        # rl4_3 = Reshape((-1, 4))(rl4_3)
        # rl4_3 = Activation("sigmoid")(rl4_3)
        
        # # Classification cl7_1
        # cl7_1 = Conv2D(4*1, (3, 3), padding='same')(conv7_1)
        # cl7_1 = Reshape((-1, 1))(cl7_1)
        # cl7_1 = Activation('sigmoid')(cl7_1)
        # # Regression cl7_1
        # rl7_1 = Conv2D(4 * 4, (3,3), padding='same')(conv7_1)
        # rl7_1 = Reshape((-1, 4))(rl7_1)
        # rl7_1 = Activation('sigmoid')(rl7_1)
        
        # c_layer = tf.concat([cl4_3, cl7_1], axis=1) # type: ignore
        # r_layer = tf.concat([rl4_3, rl7_1], axis=1)  # type: ignore
        
        f1 = GlobalMaxPool2D()(vgg)
        cl1 = Dense(2048, activation="relu")(f1)
        cl2 = Dense(1, activation="sigmoid")(cl1)
        rl1 = Dense(1024, activation='relu')(f1)
        rl2 = Dense(4, activation="sigmoid")(rl1)
        model = Model(inputs=vgg.input, outputs=[cl2, rl2])
        # print(model.summary())
        return model

    def optimizer(self, len_data, lr=0.0001, opt_type="adam"):
        lr_decay = (1.0 / 0.75 - 1) / len_data
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        return opt

    def localization_loss(self, y_true, yhat):
        delta_coor = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))
        h_true = y_true[:, 3] - y_true[:, 1]
        w_true = y_true[:, 2] - y_true[:, 0]
        h_pred = yhat[:, 3] - yhat[:, 1]
        w_pred = yhat[:, 2] - yhat[:, 0]
        delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))  # type: ignore
        regress_loss = tf.cast(delta_coor, dtype=tf.float32) + tf.cast(delta_size, dtype=tf.float32)  # type: ignore
        return regress_loss

    def classification_loss(self):
        class_loss = tf.keras.losses.BinaryCrossentropy()
        return class_loss


if __name__ == "__main__":
    _instance_class = ModelSkripsi()
    _instance_img_load = img_load.ImgLoad()
    _instance_data_load = data_load.DataLoad()

    _instance_img_load._limit_gpu()

    train, test = _instance_data_load.load()
    # model = _instance_class.build_model()
    
    _instance_custom_model = custom_model.CustomModel(input_shape=(300, 300, 3))
    # _instance_custom_model.build((None, 120, 120, 3))
    input_tensor = tf.keras.Input(shape=(300, 300, 3))
    # model_output = _instance_custom_model.call(input_tensor=input_tensor)
    # models = Model(inputs=input_tensor, outputs=model_output)
    print('sebelum')
    _instance_custom_model(input_tensor)

    _instance_custom_model.summary()
    # models.summary()
    

    X, y = train.as_numpy_iterator().next()
    # print(X.shape)
    # clss, crs = model.predict(X)
    classes, coors = _instance_custom_model.predict(X)
    # classes, coors = model.predict(X)
    print(classes[0])
    # print(classes.shape)
    # print(coors)
    
    optimizer = _instance_class.optimizer(len(train))
    local_loss = _instance_class.localization_loss
    class_loss = _instance_class.classification_loss()
    # print(train)
    # print(len(train))

    # print(y)
    # print(coors)
    # print(local_loss(y[1], coors))
    # print("==========")
    # print(class_loss(y[0], classes))
    # _instance_custom_model.compile(optimizer, class_loss, local_loss)
    # logdir = "logs"
    # tensorboard_callback = TensorBoard(log_dir=logdir)
    # _instance_custom_model.compute_output_shape(input_shape=(None, 120, 120, 3))
    # _instance_custom_model.fit(train, epochs=5, callbacks=[tensorboard_callback])
    # _instance_custom_model.save("model", save_format="tf")
    # # print(hist.history)
