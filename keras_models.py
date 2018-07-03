from inspect import getmembers, isfunction
import keras
import numpy as np

from keras.applications.vgg19 import VGG19
from keras.backend import tensorflow_backend as K
from keras.layers import BatchNormalization, Dense, Flatten, Input
from keras.layers import Conv2D, ConvLSTM2D, GRU, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, SeparableConv2D
from keras.layers import Bidirectional, Lambda, TimeDistributed
from keras.layers import Activation, Add, Average, Concatenate, Dropout, Maximum, Multiply, Permute, Reshape
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


def VGG19_AttentionLSTM(
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


def VGG19_GRU(
        frames_input_shape,
        poses_input_shape,
        classes):
    frames = Input(shape=frames_input_shape, name='frames')
    poses = Input(poses_input_shape, name='poses')

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
    
    pose_gru = GRU(512, return_sequences=True, 
                   recurrent_dropout=0.2, dropout=0.2)
    gru_1 = GRU(512, return_sequences=True, 
                recurrent_dropout=0.2, dropout=0.2)
    gru_2 = GRU(512, recurrent_dropout=0.2)    
    
    frames_features = gru_1(frames_features)
    frames_features = gru_2(frames_features)
    frames_outputs = Dense(classes, activation='softmax', name='frame_pred')(frames_features)

    poses_features = pose_gru(poses)
    poses_features = gru_1(poses_features)
    poses_features = gru_2(poses_features)
    poses_outputs = Dense(classes, activation='softmax', name='pose_pred')(poses_features)

    outputs = Average(name='avg_fusion')([frames_outputs, poses_outputs])
    
    model = Model(inputs=[frames, poses], outputs=outputs)
    # Overload model's weights with the pre-trained ImageNet weights of VGG19
    vgg19 = VGG19(include_top=False, input_shape=frames_input_shape[1:])
    for i, layer in enumerate(vgg19.layers[:-1]):
        model.layers[i].set_weights(weights=layer.get_weights())
        model.layers[i].trainable = False

    return model


def TSNs_SpatialStream(
    input_shape, classes, num_segments=3, base_model='Xception', dropout_prob=0.8, consensus_type='avg', partial_bn=True):
    """
    Spatial stream of the Temporal Segment Networks (https://arxiv.org/pdf/1705.02953.pdf) defined as multi-input Keras model.
    """
    # Define the shared layers, base conv net and enable partial batch normalization strategy
    inputs = [Input(input_shape) for _ in range(num_segments)]
    dropout = Dropout(dropout_prob)
    dense = Dense(classes, activation=None)
    act = Activation(activation='softmax', name='prediction')
    models_dict = dict(getmembers(keras.applications, isfunction))
    base = models_dict[base_model](include_top=False, pooling='avg')
    if partial_bn:
        num_bn_layers = 0
        for layer in base.layers:
            if isinstance(layer, BatchNormalization):
                num_bn_layers += 1
                if num_bn_layers != 1:
                    layer.trainable = False
    # Pass multiple inputs (depending on num_segments) through the base conv net
    outputs = []
    visual_features = []
    for seg_input in inputs:
        seg_output = base(seg_input)
        visual_features.append(seg_output)
        seg_output = dropout(seg_output)
        seg_output = dense(seg_output)
        outputs.append(seg_output)
    # Use a consensus function to combine class scores
    if consensus_type == 'avg':
        output = Average()(outputs)
    elif consensus_type == 'max':
        output = Maximum()(outputs)
    elif consensus_type == 'attention':
        weighted_outputs = []
        attn_layer = Dense(1, use_bias=False, name='attn_layer')
        attn_weights = [attn_layer(_) for _ in visual_features]
        attn_weights = Lambda(lambda x: K.concatenate(x, axis=-1), name='concatenate')(attn_weights)
        attn_weights = Activation('softmax')(attn_weights)
        for i, seg_output in enumerate(outputs):
            weight = Lambda(lambda x: x[:, i])(attn_weights)
            weighted_outputs.append(Multiply()([weight, seg_output]))
        output = Add()(weighted_outputs)         

    output = act(output)
    model = Model(inputs, output)
    return model


def TSNs_MotionStream(
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

    first_conv_weights = np.average(first_conv_weights, axis=2)
    first_conv_weights = np.reshape(
        first_conv_weights,
        (first_conv_weights.shape[0],
         first_conv_weights.shape[1],
         1,
         first_conv_weights.shape[2]))
    first_conv_weights = np.dstack([first_conv_weights] * input_shape[2])
    model.layers[1].set_weights([first_conv_weights])
    if partial_bn:
        num_bn_layers = 0
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                num_bn_layers += 1
                if num_bn_layers != 1:
                    layer.trainable = False

    return model