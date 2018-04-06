from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GRU, TimeDistributed, Bidirectional, Concatenate
from keras.applications.vgg19 import VGG19


def MultiTask_VGG19_SpatialMotionTemporalGRU(frames_input_shape,
                                             newvids_classes,
                                             yt8mvids_classes,
                                             finetune_conv_layers=False):
    newvids_frames = Input(shape=frames_input_shape, name='newvids_frames')
    yt8mvids_frames = Input(shape=frames_input_shape, name='yt8mvids_frames')
    # Block 1
    block1_conv1 = TimeDistributed(
        Conv2D(
            64,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block1_conv1'))
    block1_conv2 = TimeDistributed(
        Conv2D(
            64,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block1_conv2'))
    block1_pool = TimeDistributed(
        MaxPooling2D(
            (2, 2), strides=(
                2, 2), name='block1_pool'))

    # Block 2
    block2_conv1 = TimeDistributed(
        Conv2D(
            128,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block2_conv1'))
    block2_conv2 = TimeDistributed(
        Conv2D(
            128,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block2_conv2'))
    block2_pool = TimeDistributed(
        MaxPooling2D(
            (2, 2), strides=(
                2, 2), name='block2_pool'))

    # Block 3
    block3_conv1 = TimeDistributed(
        Conv2D(
            256,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block3_conv1'))
    block3_conv2 = TimeDistributed(
        Conv2D(
            256,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block3_conv2'))
    block3_conv3 = TimeDistributed(
        Conv2D(
            256,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block3_conv3'))
    block3_conv4 = TimeDistributed(
        Conv2D(
            256,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block3_conv4'))
    block3_pool = TimeDistributed(
        MaxPooling2D(
            (2, 2), strides=(
                2, 2), name='block3_pool'))

    # Block 4
    block4_conv1 = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block4_conv1'))
    block4_conv2 = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block4_conv2'))
    block4_conv3 = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block4_conv3'))
    block4_conv4 = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block4_conv4'))
    block4_pool = TimeDistributed(
        MaxPooling2D(
            (2, 2), strides=(
                2, 2), name='block4_pool'))

    # Block 5
    block5_conv1 = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block5_conv1'))
    block5_conv2 = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block5_conv2'))
    block5_conv3 = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block5_conv3'))
    block5_conv4 = TimeDistributed(
        Conv2D(
            512,
            (3,
             3),
            activation='relu',
            padding='same',
            name='block5_conv4'))
    global_maxpool = TimeDistributed(GlobalMaxPooling2D())

    gru1 = GRU(512, return_sequences=True, recurrent_dropout=0.2,
               dropout=0.2, name='gru1')
    gru2 = GRU(512, recurrent_dropout=0.2, name='gru2')
    newvids_dense = Dense(
        newvids_classes,
        activation='softmax',
        name='newvids_predictions')
    yt8mvids_dense = Dense(
        yt8mvids_classes,
        activation='sigmoid',
        name='yt8mvids_predictions')

    newvids_frames_features = block1_conv1(newvids_frames)
    newvids_frames_features = block1_conv2(newvids_frames_features)
    newvids_frames_features = block1_pool(newvids_frames_features)
    newvids_frames_features = block2_conv1(newvids_frames_features)
    newvids_frames_features = block2_conv2(newvids_frames_features)
    newvids_frames_features = block2_pool(newvids_frames_features)
    newvids_frames_features = block3_conv1(newvids_frames_features)
    newvids_frames_features = block3_conv2(newvids_frames_features)
    newvids_frames_features = block3_conv3(newvids_frames_features)
    newvids_frames_features = block3_conv4(newvids_frames_features)
    newvids_frames_features = block3_pool(newvids_frames_features)
    newvids_frames_features = block4_conv1(newvids_frames_features)
    newvids_frames_features = block4_conv2(newvids_frames_features)
    newvids_frames_features = block4_conv3(newvids_frames_features)
    newvids_frames_features = block4_conv4(newvids_frames_features)
    newvids_frames_features = block4_pool(newvids_frames_features)
    newvids_frames_features = block5_conv1(newvids_frames_features)
    newvids_frames_features = block5_conv2(newvids_frames_features)
    newvids_frames_features = block5_conv3(newvids_frames_features)
    newvids_frames_features = block5_conv4(newvids_frames_features)
    newvids_frames_features = global_maxpool(newvids_frames_features)
    newvids_frames_features = gru1(newvids_frames_features)
    newvids_frames_features = gru2(newvids_frames_features)
    newvids_outputs = newvids_dense(newvids_frames_features)

    yt8mvids_frames_features = block1_conv1(yt8mvids_frames)
    yt8mvids_frames_features = block1_conv2(yt8mvids_frames_features)
    yt8mvids_frames_features = block1_pool(yt8mvids_frames_features)
    yt8mvids_frames_features = block2_conv1(yt8mvids_frames_features)
    yt8mvids_frames_features = block2_conv2(yt8mvids_frames_features)
    yt8mvids_frames_features = block2_pool(yt8mvids_frames_features)
    yt8mvids_frames_features = block3_conv1(yt8mvids_frames_features)
    yt8mvids_frames_features = block3_conv2(yt8mvids_frames_features)
    yt8mvids_frames_features = block3_conv3(yt8mvids_frames_features)
    yt8mvids_frames_features = block3_conv4(yt8mvids_frames_features)
    yt8mvids_frames_features = block3_pool(yt8mvids_frames_features)
    yt8mvids_frames_features = block4_conv1(yt8mvids_frames_features)
    yt8mvids_frames_features = block4_conv2(yt8mvids_frames_features)
    yt8mvids_frames_features = block4_conv3(yt8mvids_frames_features)
    yt8mvids_frames_features = block4_conv4(yt8mvids_frames_features)
    yt8mvids_frames_features = block4_pool(yt8mvids_frames_features)
    yt8mvids_frames_features = block5_conv1(yt8mvids_frames_features)
    yt8mvids_frames_features = block5_conv2(yt8mvids_frames_features)
    yt8mvids_frames_features = block5_conv3(yt8mvids_frames_features)
    yt8mvids_frames_features = block5_conv4(yt8mvids_frames_features)
    yt8mvids_frames_features = global_maxpool(yt8mvids_frames_features)
    yt8mvids_frames_features = gru1(yt8mvids_frames_features)
    yt8mvids_frames_features = gru2(yt8mvids_frames_features)
    yt8mvids_outputs = yt8mvids_dense(yt8mvids_frames_features)

    model = Model(
        inputs=[
            newvids_frames, yt8mvids_frames], outputs=[
            newvids_outputs, yt8mvids_outputs])

    # Overload model's weights with the pre-trained ImageNet weights of VGG19
    vgg19 = VGG19(include_top=False, input_shape=frames_input_shape[1:])
    for i, layer in enumerate(vgg19.layers[:-1]):
        model.layers[i + 1].set_weights(weights=layer.get_weights())
        model.layers[i + 1].trainable = False

    # Finetune the last 2 convolutional layers in block 5 of VGG19
    if finetune_conv_layers:
        model.layers[-6].trainable = True
        model.layers[-7].trainable = True

    return model


if __name__ == '__main__':
    model = MultiTask_VGG19_SpatialMotionTemporalGRU(frames_input_shape=(
        32, 224, 224, 3), newvids_classes=7, yt8mvids_classes=4716)
    model.summary()
