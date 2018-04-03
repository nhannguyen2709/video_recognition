from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GRU, TimeDistributed, Concatenate
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19


def MotionTemporalGRU(poses_input_shape, classes):
    poses = Input(shape=poses_input_shape, name='poses')
    poses_features = GRU(256, return_sequences=True, recurrent_dropout=0.2,
                          dropout=0.2, name='gru1')(poses)
    poses_features = GRU(128, recurrent_dropout=0.2,
                          name='gru2')(poses_features)
    outputs = Dense(classes, activation='softmax',
                    name='predictions')(poses_features)
    model = Model(inputs=poses, outputs=outputs)

    return model


def VGG16_SpatialTemporalGRU(frames_input_shape, classes):
    frames = Input(shape=frames_input_shape, name='frames')
    # Block 1
    frames_features = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(frames)
    frames_features = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(frames_features)   
    
    # Block 2
    frames_features = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(frames_features)
    frames_features = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(frames_features)  
    
    # Block 3
    frames_features = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(frames_features)
    frames_features = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(frames_features)
    frames_features = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(frames_features)   
   
    # Block 4
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(frames_features)   
    
    # Block 5
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(frames_features)
    frames_features = TimeDistributed(GlobalMaxPooling2D())(frames_features)  
    
    frames_features = GRU(256, return_sequences=True, recurrent_dropout=0.2,
                          dropout=0.2, name='gru1')(frames_features)
    frames_features = GRU(128, recurrent_dropout=0.2,
                          name='gru2')(frames_features)
    outputs = Dense(classes, activation='softmax',
                    name='predictions')(frames_features)
    model = Model(inputs=frames_features, outputs=outputs)

    # Overload model's weights with the pre-trained ImageNet weights of VGG19
    vgg16 = VGG16(include_top=False, input_shape=frames_input_shape[1:])
    for i, layer in enumerate(vgg16.layers[:-1]):
        model.layers[i].set_weights(weights=layer.get_weights())
        model.layers[i].trainable = False  

    return model


def VGG19_SpatialTemporalGRU(frames_input_shape, classes):
    frames = Input(shape=frames_input_shape, name='frames')
    # Block 1
    frames_features = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(frames)
    frames_features = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(frames_features)   
    
    # Block 2
    frames_features = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(frames_features)
    frames_features = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(frames_features)  
    
    # Block 3
    frames_features = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(frames_features)
    frames_features = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(frames_features)
    frames_features = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(frames_features)
    frames_features = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(frames_features)   
   
    # Block 4
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'))(frames_features)
    frames_features = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(frames_features)   
    
    # Block 5
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(frames_features)
    frames_features = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'))(frames_features)
    frames_features = TimeDistributed(GlobalMaxPooling2D())(frames_features)  
    
    frames_features = GRU(256, return_sequences=True, recurrent_dropout=0.2,
                          dropout=0.2, name='gru1')(frames_features)
    frames_features = GRU(128, recurrent_dropout=0.2,
                          name='gru2')(frames_features)
    outputs = Dense(classes, activation='softmax', name='predictions')(frames_features)
    model = Model(inputs=frames, outputs=outputs)  

    # Overload model's weights with the pre-trained ImageNet weights of VGG19
    vgg19 = VGG19(include_top=False, input_shape=frames_input_shape[1:])
    for i, layer in enumerate(vgg19.layers[:-1]):
        model.layers[i].set_weights(weights=layer.get_weights())
        model.layers[i].trainable = False  

    return model


if __name__=='__main__':
    pass