import numpy as np

from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.layers import BatchNormalization, Dense, Flatten, Input
from keras.layers import Conv2D, ConvLSTM2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, SeparableConv2D
from keras.layers import Bidirectional, GRU, TimeDistributed
from keras.layers import Activation, Add, Average, Dropout, Multiply
from keras.models import Model, load_model
from keras.utils import multi_gpu_model


class MultiGPUModel(Model):
    def __init__(self, base_model, gpus):
        parallel_model = multi_gpu_model(base_model, gpus)
        self.__dict__.update(parallel_model.__dict__)
        self._base_model = base_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the base model. The
        base model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._base_model, attrname)

        return super(MultiGPUModel, self).__getattribute__(attrname)


def VGG19_Attention_LSTM(
        frames_input_shape,
        classes):
    frames = Input(shape=frames_input_shape, name='frames')
    vgg19 = VGG19(include_top=False)
    for layer in vgg19.layers:
        layer.trainable = False

    frames_features = TimeDistributed(vgg19)(frames)
    hidden_states = ConvLSTM2D(512, (3, 3), padding='same',
                               return_sequences=True, recurrent_dropout=0.2, dropout=0.2, name='conv_lstm_1')(frames_features)
    convolved_hidden_states = TimeDistributed(
        Conv2D(
            512,
            (1,
             1),
            activation='relu',
            padding='same',
            name='hidden_conv'))(hidden_states)
    convolved_frames_features = TimeDistributed(
        Conv2D(
            512,
            (1,
             1),
            activation='relu',
            padding='same',
            name='frames_conv'))(frames_features)
    z = TimeDistributed(
        Conv2D(
            1,
            (1,
             1),
            activation='relu',
            padding='same',
            name='attention_conv'))(Activation(activation='tanh')(Add()([convolved_frames_features, convolved_hidden_states])))
    attention = Activation(activation='softmax')(z)
    x_tilda = Multiply()([frames_features, attention])
    x_tilda = ConvLSTM2D(512, (3, 3), padding='same',
                         recurrent_dropout=0.2, dropout=0.2, name='conv_lstm_2')(x_tilda)

    x_tilda = GlobalMaxPooling2D(name='max_pool')(x_tilda)
    outputs = Dense(classes, activation='softmax',
                    name='predictions')(x_tilda)
    model = Model(inputs=frames, outputs=outputs)

    return model


def VGG19_SpatialMotionTemporalGRU(
        frames_input_shape,
        poses_input_shape,
        classes):
    frames = Input(shape=frames_input_shape, name='frames')
    # Block 1
    frames_features = TimeDistributed(
        Conv2D(
            64,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block1_conv1'))(frames)
    frames_features = TimeDistributed(
        Conv2D(
            64,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block1_conv2'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D(
        (2, 2), strides=(2, 2), name='block1_pool'))(frames_features)

    # Block 2
    frames_features = TimeDistributed(
        Conv2D(
            128,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block2_conv1'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            128,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block2_conv2'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D(
        (2, 2), strides=(2, 2), name='block2_pool'))(frames_features)

    # Block 3
    frames_features = TimeDistributed(
        Conv2D(
            256,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block3_conv1'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            256,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block3_conv2'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            256,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block3_conv3'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            256,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block3_conv4'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D(
        (2, 2), strides=(2, 2), name='block3_pool'))(frames_features)

    # Block 4
    frames_features = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block4_conv1'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block4_conv2'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block4_conv3'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block4_conv4'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D(
        (2, 2), strides=(2, 2), name='block4_pool'))(frames_features)

    # Block 5
    frames_features = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block5_conv1'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block5_conv2'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block5_conv3'))(frames_features)
    frames_features = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block5_conv4'))(frames_features)
    frames_features = TimeDistributed(GlobalMaxPooling2D())(frames_features)

    pose_gru = GRU(512, return_sequences=True, recurrent_dropout=0.2,
                   dropout=0.2, name='pose_gru')
    gru1 = GRU(512, return_sequences=True, recurrent_dropout=0.2,
               dropout=0.2, name='gru1')
    gru2 = GRU(512, recurrent_dropout=0.2, name='gru2')

    frames_features = gru1(frames_features)
    frames_features = gru2(frames_features)
    frames_outputs = Dense(classes, activation='softmax',
                           name='frames_predictions')(frames_features)

    poses = Input(poses_input_shape, name='poses')
    poses_features = pose_gru(poses)
    poses_features = gru1(poses_features)
    poses_features = gru2(poses_features)
    poses_outputs = Dense(classes, activation='softmax',
                          name='poses_predictions')(poses_features)

    outputs = Average(name='avg_fusion')([frames_outputs, poses_outputs])
    model = Model(inputs=[frames, poses], outputs=outputs)

    return model


def TemporalSegmentNetworks_SpatialStream(
        input_shape, dropout_prob, classes, partial_bn=True):
    img_input_1 = Input(shape=input_shape)
    img_input_2 = Input(shape=input_shape)
    img_input_3 = Input(shape=input_shape)

    xception = Xception(include_top=False, pooling='avg')
    # partial batch-normalization
    if partial_bn:
        num_bn_layers = 0
        for layer in xception.layers:
            if isinstance(layer, BatchNormalization):
                num_bn_layers += 1
                if num_bn_layers != 1:
                    layer.trainable = False
    dense_no_act = Dense(classes, activation=None, name='dense_no_act')
    dropout = Dropout(dropout_prob)
    x_1 = xception(img_input_1)
    x_1 = dropout(x_1)
    x_1 = dense_no_act(x_1)
    x_2 = xception(img_input_2)
    x_2 = dropout(x_2)
    x_2 = dense_no_act(x_2)
    x_3 = xception(img_input_3)
    x_3 = dropout(x_3)
    x_3 = dense_no_act(x_3)
    x = Average(name='agg_func')([x_1, x_2, x_3])
    x = Activation(activation='softmax', name='predictions')(x)
    model = Model(inputs=[img_input_1, img_input_2, img_input_3], outputs=x)

    return model


def TemporalSegmentNetworks_MotionStream(
        input_shape, dropout_prob, classes, weights='spatial_stream', partial_bn=True):
    flow_input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False,
               name='block1_conv1')(flow_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same', name='block2_pool')(x)
    x = Add()([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same',
                        use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same',
                        use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same', name='block3_pool')(x)
    x = Add()([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same',
                        use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same',
                        use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same', name='block4_pool')(x)
    x = Add()([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same',
                            use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same',
                            use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same',
                            use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = Add()([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same',
                        use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same',
                        use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same', name='block13_pool')(x)
    x = Add()([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same',
                        use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same',
                        use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(dropout_prob)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=flow_input, outputs=x)

    if weights == 'imagenet':
        xception = Xception(include_top=False, pooling='avg')
        for old_layer, new_layer in zip(xception.layers[2:], model.layers[2:]):
            new_layer.set_weights(old_layer.get_weights())
        first_conv_weights = xception.layers[1].get_weights()[0]

    elif weights == 'spatial_stream':
        xception = load_model('checkpoint/ucf101_spatial_stream.hdf5')
        for old_layer, new_layer in zip(xception.layers[4:], model.layers[2:]):
            new_layer.set_weights(old_layer.get_weights())
        first_conv_weights = xception.layers[3].get_weights()[0]

    # cross-modality pre-training
    first_conv_weights = np.average(first_conv_weights, axis=2)
    first_conv_weights = np.reshape(
        first_conv_weights,
        (first_conv_weights.shape[0],
         first_conv_weights.shape[1],
         1,
         first_conv_weights.shape[2]))
    first_conv_weights = np.dstack([first_conv_weights] * input_shape[2])
    model.layers[1].set_weights([first_conv_weights])
    # partial batch-normalization
    if partial_bn:
        num_bn_layers = 0
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                num_bn_layers += 1
                if num_bn_layers != 1:
                    layer.trainable = False

    return model


if __name__ == '__main__':
    model = VGG19_SpatialMotionTemporalGRU(frames_input_shape=(None, 224, 224, 3),
                                           poses_input_shape=(None, 26), classes=5)
    model.summary()
    model = VGG19_Attention_LSTM(
        frames_input_shape=(
            None, 224, 224, 3), classes=15)
    model.summary()
    model = TemporalSegmentNetworks_SpatialStream(
        input_shape=(299, 299, 3), dropout_prob=0.8, classes=101)
    model.summary()
    model = TemporalSegmentNetworks_MotionStream(
        input_shape=(299, 299, 20), dropout_prob=0.7, classes=101)
    model.summary()
