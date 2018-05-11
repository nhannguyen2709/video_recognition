from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv2D, ConvLSTM2D, MaxPooling2D, GlobalMaxPooling2D, GRU, TimeDistributed, Bidirectional
from keras.layers import Activation, Add, Average, Multiply
from keras.models import Model
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
        classes,
        finetune_conv_layers=False):
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

    # Overload model's weights with the pre-trained ImageNet weights of VGG19
    vgg19 = VGG19(include_top=False, input_shape=frames_input_shape[1:])
    for i, layer in enumerate(vgg19.layers[:-1]):
        model.layers[i].set_weights(weights=layer.get_weights())
        model.layers[i].trainable = False

    # Finetune the last 2 convolutional layers in block 5 of VGG19
    if finetune_conv_layers:
        model.layers[-12].trainable = True
        model.layers[-13].trainable = True

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

    # Overload model's weights with the pre-trained ImageNet weights of VGG19
    vgg19 = VGG19(include_top=False, input_shape=frames_input_shape[1:])
    for i, layer in enumerate(vgg19.layers[:-1]):
        model.layers[i].set_weights(weights=layer.get_weights())
        model.layers[i].trainable = False

    return model


if __name__ == '__main__':
    # model = VGG19_SpatialMotionTemporalGRU(frames_input_shape=(32, 224, 224, 3), 
    #                                        poses_input_shape=(32, 26), classes=3)
    # model.summary()
    model = VGG19_Attention_LSTM(frames_input_shape=(None, 224, 224, 3), classes=15, finetune_conv_layers=True)
    model.summary()
