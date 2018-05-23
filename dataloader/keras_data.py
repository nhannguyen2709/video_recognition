import gc
import json
import os
import numpy as np
import pandas as pd
import pickle
import random
from scipy import io

from cv2 import resize, imread
from sklearn.utils import shuffle
from keras.utils import Sequence, to_categorical


class UCF101Frames(Sequence):
    def __init__(self, frames_path, batch_size, shuffle=True):
        self.frames_path = frames_path
        self.get_video_frames_paths_and_labels()
        print('Found {} videos belonging to {} classes'.format(len(self.x), len(self.labels)))
        self.batch_size = batch_size
        self.num_classes = len(self.labels)
        self.shuffle = shuffle
        self.on_train_begin()
        self.on_epoch_end()

    def on_train_begin(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)

    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)

    def get_video_frames_paths_and_labels(self):
        videos = sorted(os.listdir(self.frames_path))
        self.x = [os.path.join(self.frames_path, video)
                  for video in videos]
        self.labels = sorted(set([video.split('_')[1] for video in videos]))
        self.y = []
        for video in videos:
            self.y.append(self.labels.index(video.split('_')[1]))

    def sample_frames(self, video_path):
        all_frames = np.array([filename for filename in sorted(
            os.listdir(video_path)) if filename.endswith('.jpg')])
        segment1_sampled_idx = np.random.randint(int(len(all_frames) / 3))
        segment2_sampled_idx = np.random.randint(int(len(all_frames) / 3), int(len(all_frames) * 2 / 3))
        segment3_sampled_idx = np.random.randint(int(len(all_frames) * 2 / 3), len(all_frames))

        return all_frames[segment1_sampled_idx], all_frames[segment2_sampled_idx], all_frames[segment3_sampled_idx]
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_segment1_frames = np.zeros((len(batch_x), 299, 299, 3))
        batch_segment2_frames = np.zeros((len(batch_x), 299, 299, 3))
        batch_segment3_frames = np.zeros((len(batch_x), 299, 299, 3))
        
        for i, video_path in enumerate(batch_x):
            segment1_sampled_frame, segment2_sampled_frame, segment3_sampled_frame = self.sample_frames(video_path)
            batch_segment1_frames[i] = resize(imread(os.path.join(video_path, segment1_sampled_frame)), (299, 299))
            batch_segment2_frames[i] = resize(imread(os.path.join(video_path, segment2_sampled_frame)), (299, 299))
            batch_segment3_frames[i] = resize(imread(os.path.join(video_path, segment3_sampled_frame)), (299, 299))

        batch_segment1_frames /= 255.
        batch_segment2_frames /= 255.
        batch_segment3_frames /= 255.

        return [batch_segment1_frames, batch_segment2_frames, batch_segment3_frames], to_categorical(
            np.array(batch_y), num_classes=self.num_classes)

    
class MyVideos(Sequence):
    def __init__(self, frames_path, poses_path, batch_size,
                 num_frames_sampled, shuffle=True):
        self.frames_path = frames_path
        self.poses_path = poses_path
        self.get_video_frames_poses_paths_and_labels()
        print('Found {} videos belonging to {} classes'.format(len(self.x), len(self.labels)))
        self.batch_size = batch_size
        self.num_frames_sampled = num_frames_sampled
        self.num_classes = len(self.labels)
        self.shuffle = shuffle
        self.on_train_begin()
        self.on_epoch_end()

    def on_train_begin(self):
        if self.shuffle:
            self.x, self.y, self.p = shuffle(
                self.x, self.y, self.p)

    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y, self.p = shuffle(
                self.x, self.y, self.p)

    def get_video_frames_poses_paths_and_labels(self):
        videos = sorted(os.listdir(self.frames_path))
        self.x = [os.path.join(self.frames_path, video)
                  for video in videos]
        self.p = [os.path.join(self.poses_path, video)
                  for video in videos]
        self.labels = sorted(set([video.split('_')[0] for video in videos]))
        self.y = []
        for video in videos:
            self.y.append(self.labels.index(video.split('_')[0]))
    
    def sample_frames(self, video_path, video_poses_path):
        all_frames = np.array([filename for filename in sorted(
            os.listdir(video_path)) if filename.endswith('.jpg')])
        all_poses = np.array([filename for filename in sorted(
            os.listdir(video_poses_path)) if filename.endswith('.json')])
        sampled_frames_idx = sorted(np.random.choice(
            len(all_frames), size=int(self.num_frames_sampled), replace=False))
        return all_frames[sampled_frames_idx], all_poses[sampled_frames_idx]
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_p = self.p[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_video_frames = np.zeros(
            (len(batch_x), self.num_frames_sampled, 224, 224, 3))
        batch_video_poses = np.zeros(
            (len(batch_x), self.num_frames_sampled, 26))

        i = 0
        for video_path, video_poses_path in zip(batch_x, batch_p):
            sampled_frames, sampled_poses = self.sample_frames(
                video_path, video_poses_path)
            # extract video frames
            batch_video_frames[i] = [
                resize(
                    imread(
                        os.path.join(
                            video_path,
                            frame)),
                    (224,
                     224)) for frame in sampled_frames]
            # extract poses from sampled frames
            video_poses = []
            pose_body_parts = [1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10] # match with PennAction dataset
            first_frame = imread(os.path.join(video_path, sampled_frames[0]))
            x_max, y_max = first_frame.shape[:2]
            for pose in sampled_poses:
                with open(os.path.join(video_poses_path, pose)) as json_data:
                    json_file = json.load(json_data)
                    pose_keypoints = json_file['people'][0]['pose_keypoints_2d']
                    poses_x = np.array(pose_keypoints[::3])[pose_body_parts]
                    poses_y = np.array(pose_keypoints[1::3])[pose_body_parts]
                    poses_x, poses_y = poses_x / x_max, poses_y / y_max
                    poses_x_y = np.hstack([poses_x, poses_y])
                    video_poses.append(poses_x_y)
            batch_video_poses[i] = video_poses
            i += 1

        batch_video_frames /= 255.

        return [batch_video_frames, batch_video_poses], to_categorical(
            np.array(batch_y), num_classes=self.num_classes)


class PennAction(Sequence):
    def __init__(self, frames_path, labels_path, batch_size,
                 num_frames_sampled, shuffle=True):
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.get_videos_paths()
        self.extract_mat_file()
        self.num_classes = len(self.labels)
        print('Found {} videos belonging to {} classes'.format(len(self.x), self.num_classes))
        self.batch_size = batch_size
        self.num_frames_sampled = num_frames_sampled
        self.shuffle = shuffle
        self.on_train_begin()
        self.on_epoch_end()

    def on_train_begin(self):
        if self.shuffle:
            self.x, self.y, self.frame_counts = shuffle(
                self.x, self.y, self.frame_counts)

    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y, self.frame_counts = shuffle(
                self.x, self.y, self.frame_counts)

    def get_videos_paths(self):
        videos = sorted(os.listdir(self.frames_path))
        self.x = [os.path.join(self.frames_path, video)
                  for video in videos]

    def extract_mat_file(self):
        list_mat_files = sorted(os.listdir(self.labels_path))
        y = np.empty(len(list_mat_files), dtype=object)
        frame_counts = np.empty(len(list_mat_files), dtype=int)
        for i, mat_file in enumerate(list_mat_files):
            mat = io.loadmat(os.path.join(self.labels_path, mat_file))
            frame_counts[i] = mat['nframes'][0][0]
            y[i] = mat['action'][0]
            
        self.frame_counts = frame_counts
        self.labels = sorted(set(y))
        self.y = np.vectorize(lambda x: self.labels.index(x))(
            y)  # convert label into class

    def sample_frames(self, video_path, frame_count):
        all_frames = np.array([filename for filename in sorted(
            os.listdir(video_path)) if filename.endswith('.jpg')])
        sampled_frames_idx = sorted(np.random.choice(
            frame_count - 1, size=int(self.num_frames_sampled), replace=False))
        return sampled_frames_idx, all_frames[sampled_frames_idx]
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mat_files = [video_path.replace('frames/', 'labels/') for video_path in batch_x]
        batch_frame_counts = self.frame_counts[idx * \
            self.batch_size:(idx + 1) * self.batch_size]
        batch_video_frames = np.zeros(
            (len(batch_x), self.num_frames_sampled, 224, 224, 3))
        batch_video_poses = np.zeros(
            (len(batch_x), self.num_frames_sampled, 26))

        for i, video_path in enumerate(batch_x):
            sampled_frames_idx, sampled_frames = self.sample_frames(
                video_path, batch_frame_counts[i])
            # extract video frames
            batch_video_frames[i] = [
                resize(
                    imread(
                        os.path.join(
                            video_path,
                            frame)),
                    (224,
                     224)) for frame in sampled_frames]
            # extract poses from sampled frames
            first_frame = imread(os.path.join(video_path, sampled_frames[0]))
            x_max, y_max = first_frame.shape[:2]
            mat = io.loadmat(batch_mat_files[i])
            # normalize poses coordinates
            poses_x, poses_y = mat['x'] / x_max, mat['y'] / y_max
            poses_x_y = np.hstack([poses_x, poses_y])
            poses_x_y = poses_x_y[sampled_frames_idx]
            batch_video_poses[i] = poses_x_y

        batch_video_frames /= 255.

        return [batch_video_frames, batch_video_poses], to_categorical(
            np.array(batch_y), num_classes=self.num_classes)            


if __name__=='__main__':
    import time
    
    # my_videos = MyVideos(frames_path='../data/MyVideos/frames', 
    #                      poses_path='../data/MyVideos/poses',
    #                      batch_size=4, num_frames_sampled=16,
    #                      shuffle=False)
    # print(my_videos.labels)
    # for i in range(len(my_videos)):
    #     start = time.time()
    #     x, y = my_videos.__getitem__(i)
    #     print(x[0].shape, x[1].shape, y.shape)
    #     end = time.time()
    #     print('Time taken to load a batch of {} videos: {}'.format(y.shape[0], end - start))

    ucf101_frames = UCF101Frames(frames_path='../data/UCF101/train/frames', batch_size=8, shuffle=False)
    print(ucf101_frames.labels)
    for i in range(len(ucf101_frames)):
        start = time.time()
        x, y = ucf101_frames.__getitem__(i)
        print(x[0].shape, x[1].shape, x[2].shape, y.shape)
        end = time.time()
        print('Time taken to load a batch of {} videos: {}'.format(y.shape[0], end - start))