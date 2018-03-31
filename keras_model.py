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


def VGG19_FeatureExtractor(frames_features_input_shape):
    frames = Input(shape=frames_features_input_shape, name='frames')
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
    vgg19 = VGG19(include_top=False, input_shape=frames_features_input_shape)
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


class VideosFrames(Sequence):
    def __init__(self, data_path, frame_counts_path, batch_size, num_frames_used):
        self.data_path = data_path
        self.frame_counts_path = frame_counts_path
        video_filenames = sorted(os.listdir(self.data_path))
        self.x = [os.path.join(self.data_path, video_filename) for video_filename in video_filenames] 
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
        return np.ceil(len(self.x) / float(self.batch_size)).astype(int)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_frame_counts = self.get_frame_counts_of_batch(batch_x)
        batch_video_frames = []

        for video_path, frame_counts in zip(batch_x, batch_frame_counts):        
             
            if frame_counts > self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224)) 
                                                for frame in sorted(os.listdir(video_path))[:self.num_frames_used] if frame.endswith('.jpg')])
 
            elif frame_counts < self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224)) 
                                                for frame in sorted(os.listdir(video_path))[:frame_counts] if frame.endswith('.jpg')])  
                single_video_frames = np.pad(single_video_frames, ((0, self.num_frames_used - frame_counts), (0, 0), (0, 0), (0, 0)), 
                                             mode='constant', constant_values=0)

            elif frame_counts == self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224)) 
                                                for frame in sorted(os.listdir(video_path)) if frame.endswith('.jpg')])

            batch_video_frames.append(single_video_frames)

        return np.vstack(batch_video_frames)


class VideoSequence(Sequence):
    def __init__(self, data_dir, frame_counts_path, batch_size, num_frames_used):
        self.data_dir = data_dir
        self.frame_counts_path = frame_counts_path
        video_filenames = sorted(os.listdir(self.data_dir))
        self.labels = sorted(
            list(set([video_filename.split('_')[1] for video_filename in video_filenames])))
        self.x = [os.path.join(self.data_dir, video_filename)
                  for video_filename in video_filenames]
        self.y = self.labels_to_idxs(video_filenames)
        self.batch_size = batch_size
        self.num_frames_used = num_frames_used

    def labels_to_idxs(self, video_filenames):
        idxs = []
        for video_filename in video_filenames:
            idxs.append(self.labels.index(video_filename.split('_')[1]))
        return idxs

    def get_frame_counts_of_batch(self, batch_x):
        frame_counts = pickle.load(open(self.frame_counts_path, 'rb'))
        batch_video_filenames = [video_path.split(
            '/')[-1] for video_path in batch_x]
        batch_frame_counts = [frame_counts[video_filename]
                              for video_filename in batch_video_filenames]
        return batch_frame_counts

    @staticmethod
    def extract_features(single_video_frames):
        with tf.device('/gpu:0'):
            feature_extractor = VGG19_FeatureExtractor(
                frames_features_input_shape=(224, 224, 3))
            single_video_frames_features = feature_extractor.predict(
                single_video_frames, verbose=1)
        return single_video_frames_features

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_frame_counts = self.get_frame_counts_of_batch(batch_x)
        batch_video_frames = []

        for video_path, frame_counts in zip(batch_x, batch_frame_counts):

            if frame_counts > self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224))
                                                for frame in sorted(os.listdir(video_path))[:self.num_frames_used] if frame.endswith('.jpg')])

            elif frame_counts < self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224))
                                                for frame in sorted(os.listdir(video_path))[:frame_counts] if frame.endswith('.jpg')])
                single_video_frames = np.pad(single_video_frames, ((0, self.num_frames_used - frame_counts), (0, 0), (0, 0), (0, 0)),
                                             mode='constant', constant_values=0)

            elif frame_counts == self.num_frames_used:
                single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224))
                                                for frame in sorted(os.listdir(video_path)) if frame.endswith('.jpg')])

            batch_video_frames.append(single_video_frames)

        return np.array(batch_video_frames), to_categorical(np.array(batch_y), num_classes=7)
if __name__=='__main__':
    import time
    start = time.time()
    videos_frames = VideosFrames(data_path='data/NewVideos/videos_frames/',
                                 frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                 batch_size=16, num_frames_used=250)
    batch_x = videos_frames.__getitem__(1)
    end = time.time()
    print('Time taken to load a single batch of {} videos: {}'.format(16, end - start))
    print(batch_x.shape)
    # pass
