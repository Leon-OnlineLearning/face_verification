from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model
from keras import layers
import tensorflow as tf

class Resnet50Model:
    
    def resnet_identity_block(self,input_tensor, kernel_size, filters, stage, block,
                              bias=False):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
        conv1_increase_name = 'conv' + str(stage) + "_" + str(
            block) + "_1x1_increase"
        conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

        x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
            input_tensor)
        x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, use_bias=bias,
                   padding='same', name=conv3_name)(x)
        x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
        x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def resnet_conv_block(self,input_tensor, kernel_size, filters, stage, block,
                          strides=(2, 2), bias=False):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
        conv1_increase_name = 'conv' + str(stage) + "_" + str(
            block) + "_1x1_increase"
        conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
        conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

        x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
                   name=conv1_reduce_name)(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
                   name=conv3_name)(x)
        x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
        x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                          name=conv1_proj_name)(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(
            shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def resnet50(self):
        inputs = tf.keras.Input(shape=(224,224,3))
        x = Conv2D(
            64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
            name='conv1/7x7_s2')(inputs)
        x = BatchNormalization(axis=3, name='conv1/7x7_s2/bn')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = self.resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
        x = self.resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
        x = self.resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)
        x = self.resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
        x = self.resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
        x = self.resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
        x = self.resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)
        x = self.resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
        x = self.resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
        x = self.resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
        x = self.resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
        x = self.resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
        x = self.resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)
        x = self.resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
        x = self.resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
        x = self.resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)
        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        model = Model(inputs, x, name='vggface_resnet50')
        return(model)
        


        
