import os
import numpy as np
import pickle

import tensorflow as tf
from sklearn.utils import shuffle
from cv2 import resize, imread

from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GRU, TimeDistributed, Concatenate
from keras.applications.vgg19 import VGG19
from keras.utils import Sequence, to_categorical

def TimeDistributedVGG19_GRU(frames_input_shape, poses_input_shape, classes):
    frames = Input(shape=frames_input_shape, name='frames')
    poses = Input(shape=poses_input_shape, name='poses')
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
    # GRUs block
    merged_features = Concatenate(name='concatenate')([frames_features, poses])
    merged_features = GRU(256, return_sequences=True, recurrent_dropout=0.2, dropout=0.2, name='gru1')(merged_features)
    merged_features = GRU(128, recurrent_dropout=0.2, name='gru2')(merged_features)
    outputs = Dense(classes, activation='softmax', name='predictions')(merged_features)
    model = Model(inputs=[frames, poses], outputs=outputs)  
    # Overload model's weights with the pre-trained ImageNet weights of VGG19
    vgg19 = VGG19(include_top=False, input_shape=frames_input_shape[1:])
    for i, layer in enumerate(vgg19.layers[:-1]):
        model.layers[i].set_weights(weights=layer.get_weights())
        model.layers[i].trainable = False      
    return model


def VGG19_FeatureExtractor(frames_input_shape):
    frames = Input(shape=frames_input_shape, name='frames')
    # Block 1
    frames_features = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(frames)
    frames_features = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(frames_features)
    frames_features = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(frames_features) 
    # Block 2
    frames_features = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(frames_features)
    frames_features = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(frames_features)
    frames_features = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(frames_features)
    # Block 3
    frames_features = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(frames_features)
    frames_features = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(frames_features)
    frames_features = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(frames_features)
    frames_features = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(frames_features)
    frames_features = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(frames_features)
     # Block 4
    frames_features = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(frames_features)
    frames_features = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(frames_features)
    frames_features = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(frames_features)
    frames_features = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(frames_features)
    frames_features = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(frames_features) 
    # Block 5
    frames_features = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(frames_features)
    frames_features = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(frames_features)
    frames_features = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(frames_features)
    frames_features = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(frames_features)
    frames_features = GlobalMaxPooling2D()(frames_features)   
    model = Model(inputs=frames, outputs=frames_features)
    vgg19 = VGG19(include_top=False, input_shape=frames_input_shape)
    for i, layer in enumerate(vgg19.layers[:-1]):
        model.layers[i].set_weights(weights=layer.get_weights())
        model.layers[i].trainable = False
    return model


def TemporalGRU(frames_features_input_shape, poses_input_shape, classes):
    frames_features = Input(shape=frames_features_input_shape, name='frames')
    poses = Input(shape=poses_input_shape, name='poses')
    merged_features = Concatenate(name='concatenate')([frames_features, poses])
    merged_features = GRU(256, return_sequences=True, recurrent_dropout=0.2,
                          dropout=0.2, name='gru1')(merged_features)
    merged_features = GRU(128, recurrent_dropout=0.2,
                          name='gru2')(merged_features)
    outputs = Dense(classes, activation='softmax',
                    name='predictions')(merged_features)
    model = Model(inputs=[frames_features, poses], outputs=outputs)
    return model


class VideoSequence(Sequence):
    def __init__(self, data_dir, frame_counts_path, batch_size, num_frames_used):
        self.data_dir = data_dir
        self.frame_counts_path = frame_counts_path
        video_filenames = sorted(os.listdir(self.data_dir))
        self.labels = sorted(list(set([video_filename.split('_')[1] for video_filename in video_filenames])))
        self.x_set = [os.path.join(self.data_dir, video_filename) for video_filename in video_filenames] 
        self.y_set = self.labels_to_idxs(video_filenames)
        self.x, self.y = shuffle(self.x_set, self.y_set)
        self.batch_size = batch_size
        self.num_frames_used = num_frames_used
    
    def labels_to_idxs(self, video_filenames):
        idxs = []
        for video_filename in video_filenames:
            idxs.append(self.labels.index(video_filename.split('_')[1]))
        return idxs

    def get_frame_counts_of_batch(self, batch_x):
        frame_counts = pickle.load(open(self.frame_counts_path, 'rb'))
        batch_video_filenames = [video_path.split('/')[-1] for video_path in batch_x]
        batch_frame_counts = [frame_counts[video_filename] for video_filename in batch_video_filenames]
        return batch_frame_counts

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_frame_counts = self.get_frame_counts_of_batch(batch_x)
        batch_video_frames = []
        batch_video_poses = []

        with tf.device('/cpu:0'):
            feature_extractor = VGG19_FeatureExtractor(frames_input_shape=(224, 224, 3))

        for video_path, frame_counts in zip(batch_x, batch_frame_counts):        
            
            if frame_counts > self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224)) 
                                                for frame in sorted(os.listdir(video_path))[:self.num_frames_used] if frame.endswith('.jpg')])
                single_video_frames = feature_extractor.predict(single_video_frames, batch_size=50, verbose=1)
                single_video_poses = np.load(os.path.join(video_path, 'poses.npy'))
                single_video_poses = single_video_poses[:self.num_frames_used, :, :]
                single_video_poses = single_video_poses.reshape(self.num_frames_used, -1)
                single_video_poses[np.isnan(single_video_poses)] = -1. # fill missing coordinates with -1

            elif frame_counts < self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224)) 
                                                for frame in sorted(os.listdir(video_path))[:frame_counts] if frame.endswith('.jpg')])  
                single_video_frames = np.pad(single_video_frames, ((0, self.num_frames_used - frame_counts), (0, 0), (0, 0), (0, 0)), 
                                             mode='constant', constant_values=0)
                single_video_frames = feature_extractor.predict(single_video_frames, batch_size=50, verbose=1)
                single_video_poses = np.load(os.path.join(video_path, 'poses.npy'))
                single_video_poses = np.pad(single_video_poses, ((0, self.num_frames_used - frame_counts), (0, 0), (0, 0)),
                                            mode='constant', constant_values=0)
                single_video_poses = single_video_poses.reshape(self.num_frames_used, -1)
                single_video_poses[np.isnan(single_video_poses)] = -1.

            elif frame_counts == self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224)) 
                                                for frame in sorted(os.listdir(video_path)) if frame.endswith('.jpg')])
                single_video_frames = feature_extractor.predict(single_video_frames, batch_size=50, verbose=1)
                single_video_poses = np.load(os.path.join(video_path, 'poses.npy'))
                single_video_poses = single_video_poses.reshape(frame_counts, -1)
                single_video_poses[np.isnan(single_video_poses)] = -1.
                
            batch_video_frames.append(single_video_frames)
            batch_video_poses.append(single_video_poses)

        return [np.array(batch_video_frames), np.array(batch_video_poses)], to_categorical(np.array(batch_y), num_classes=7)
    
if __name__=='__main__':
    # temporal_gru = TemporalGRU(frames_features_input_shape=(250, 512), 
    #                            poses_input_shape=(250, 54),
    #                            classes=7)
    # temporal_gru.summary()
    import time
    start = time.time()
    video_sequence = VideoSequence(data_dir='data/NewVideos/videos_frames/',
                                   frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                   batch_size=16, num_frames_used=250)
    batch_x, batch_y = video_sequence.__getitem__(1)
    end = time.time()
    print('Time taken to load a single batch of {} videos: {}'.format(16, end - start))
    print(batch_x[0].shape, batch_x[1].shape, batch_y, batch_y.shape)
    # pass
