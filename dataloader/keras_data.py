import os
import numpy as np
import pickle

from cv2 import resize, imread
from sklearn.utils import shuffle
from keras.utils import Sequence, to_categorical


class VideosFrames(Sequence):
    def __init__(self, data_path, frame_counts_path, batch_size,
                 num_frames_sampled, num_frames_skipped, num_classes, shuffle=True):
        self.data_path = data_path
        self.frame_counts_path = frame_counts_path
        self.video_filenames = sorted(os.listdir(self.data_path))
        self.labels = sorted(
            list(set([video_filename.split('_')[1] for video_filename in self.video_filenames])))
        self.x = [os.path.join(self.data_path, video_filename)
                  for video_filename in self.video_filenames]
        self.y = self.labels_to_idxs()
        self.batch_size = batch_size
        self.num_frames_sampled = num_frames_sampled
        self.num_frames_skipped = num_frames_skipped
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle == True:
            self.x, self.y = shuffle(self.x, self.y)

    def labels_to_idxs(self):
        idxs = []
        for video_filename in self.video_filenames:
            idxs.append(self.labels.index(video_filename.split('_')[1]))
        return idxs

    def sample_frames_snippet(self, video_path, frame_counts):
        frames = np.array([filename for filename in sorted(
            os.listdir(video_path)) if filename.endswith('.jpg')])
        start_frameidx_in_snippet = np.random.randint(
            0, frame_counts - self.num_frames_sampled)
        end_frameidx__in_snippet = start_frameidx_in_snippet + self.num_frames_sampled
        framesidx_snippet = range(
            start_frameidx_in_snippet, end_frameidx__in_snippet, self.num_frames_skipped)
        return frames[framesidx_snippet]

    def get_frame_counts_of_batch(self, batch_x):
        frame_counts = pickle.load(open(self.frame_counts_path, 'rb'))
        batch_video_filenames = [video_path.split(
            '/')[-1] for video_path in batch_x]
        batch_frame_counts = [frame_counts[video_filename]
                              for video_filename in batch_video_filenames]
        return batch_frame_counts

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_frame_counts = self.get_frame_counts_of_batch(batch_x)
        batch_video_frames = []

        for video_path, frame_counts in zip(batch_x, batch_frame_counts):
            frames_snippet = self.sample_frames_snippet(
                video_path, frame_counts)
            single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224))
                                            for frame in frames_snippet])
            batch_video_frames.append(single_video_frames)

        return np.array(batch_video_frames), to_categorical(
            np.array(batch_y), num_classes=self.num_classes)


class VideosPoses(Sequence):
    def __init__(self, data_path, frame_counts_path, batch_size,
                 num_frames_sampled, num_frames_skipped, num_classes, shuffle=True):
        self.data_path = data_path
        self.frame_counts_path = frame_counts_path
        self.video_filenames = sorted(os.listdir(self.data_path))
        self.labels = sorted(
            list(set([video_filename.split('_')[1] for video_filename in self.video_filenames])))
        self.x = [os.path.join(self.data_path, video_filename)
                  for video_filename in self.video_filenames]
        self.y = self.labels_to_idxs()
        self.batch_size = batch_size
        self.num_frames_sampled = num_frames_sampled
        self.num_frames_skipped = num_frames_skipped
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle == True:
            self.x, self.y = shuffle(self.x, self.y)

    def labels_to_idxs(self):
        idxs = []
        for video_filename in self.video_filenames:
            idxs.append(self.labels.index(video_filename.split('_')[1]))
        return idxs

    def sample_framesidx_snippet(self, frame_counts):
        start_frameidx_in_snippet = np.random.randint(
            0, frame_counts - self.num_frames_sampled)
        end_frameidx__in_snippet = start_frameidx_in_snippet + self.num_frames_sampled
        framesidx_snippet = range(
            start_frameidx_in_snippet, end_frameidx__in_snippet, self.num_frames_skipped)
        return framesidx_snippet

    def get_frame_counts_of_batch(self, batch_x):
        frame_counts = pickle.load(open(self.frame_counts_path, 'rb'))
        batch_video_filenames = [video_path.split(
            '/')[-1] for video_path in batch_x]
        batch_frame_counts = [frame_counts[video_filename]
                              for video_filename in batch_video_filenames]
        return batch_frame_counts

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_frame_counts = self.get_frame_counts_of_batch(batch_x)
        batch_video_poses = []

        for video_path, frame_counts in zip(batch_x, batch_frame_counts):
            framesidx_snippet = self.sample_framesidx_snippet(frame_counts)
            video_frames_poses = np.load(os.path.join(video_path, 'poses.npy'))
            video_frames_poses = np.reshape(
                video_frames_poses, (frame_counts, -1))
            video_frames_poses[np.isnan(video_frames_poses)] = -1. # fill missing pose coordinates with -1
            video_frames_poses = video_frames_poses[framesidx_snippet]
            batch_video_poses.append(video_frames_poses)

        return np.array(batch_video_poses), to_categorical(
            np.array(batch_y), num_classes=self.num_classes)


class VideosFramesPoses(Sequence):
    def __init__(self, data_path, frame_counts_path, batch_size,
                 num_frames_sampled, num_frames_skipped, num_classes, shuffle=True):
        self.data_path = data_path
        self.frame_counts_path = frame_counts_path
        self.video_filenames = sorted(os.listdir(self.data_path))
        self.labels = sorted(
            list(set([video_filename.split('_')[1] for video_filename in self.video_filenames])))
        self.x = [os.path.join(self.data_path, video_filename)
                  for video_filename in self.video_filenames]
        self.y = self.labels_to_idxs()
        self.batch_size = batch_size
        self.num_frames_sampled = num_frames_sampled
        self.num_frames_skipped = num_frames_skipped
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle == True:
            self.x, self.y = shuffle(self.x, self.y)

    def labels_to_idxs(self):
        idxs = []
        for video_filename in self.video_filenames:
            idxs.append(self.labels.index(video_filename.split('_')[1]))
        return idxs

    def sample_frames_snippet_with_idx(self, video_path, frame_counts):
        frames = np.array([filename for filename in sorted(
            os.listdir(video_path)) if filename.endswith('.jpg')])
        start_frameidx_in_snippet = np.random.randint(
            0, frame_counts - self.num_frames_sampled)
        end_frameidx__in_snippet = start_frameidx_in_snippet + self.num_frames_sampled
        framesidx_snippet = range(
            start_frameidx_in_snippet, end_frameidx__in_snippet, self.num_frames_skipped)
        return framesidx_snippet, frames[framesidx_snippet]

    def get_frame_counts_of_batch(self, batch_x):
        frame_counts = pickle.load(open(self.frame_counts_path, 'rb'))
        batch_video_filenames = [video_path.split(
            '/')[-1] for video_path in batch_x]
        batch_frame_counts = [frame_counts[video_filename]
                              for video_filename in batch_video_filenames]
        return batch_frame_counts

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_frame_counts = self.get_frame_counts_of_batch(batch_x)
        batch_video_frames = []
        batch_video_poses = []

        for video_path, frame_counts in zip(batch_x, batch_frame_counts):
            framesidx_snippet, frames_snippet = self.sample_frames_snippet_with_idx(
                video_path, frame_counts)

            single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224))
                                            for frame in frames_snippet])
            batch_video_frames.append(single_video_frames)

            video_frames_poses = np.load(os.path.join(video_path, 'poses.npy'))
            video_frames_poses = np.reshape(
                video_frames_poses, (frame_counts, -1))
            video_frames_poses[np.isnan(video_frames_poses)] = -1. # fill missing pose coordinates with -1
            video_frames_poses = video_frames_poses[framesidx_snippet]
            batch_video_poses.append(video_frames_poses)

        return [np.array(batch_video_frames), np.array(batch_video_poses)], to_categorical(
            np.array(batch_y), num_classes=self.num_classes)


if __name__ == '__main__':
    import time
    videos_frames = VideosFrames(data_path='../data/train_videos_01',
                                 frame_counts_path='dic/merged_frame_count.pickle',
                                 batch_size=8,
                                 num_classes=101,
                                 num_frames_sampled=48,
                                 num_frames_skipped=3)
    for i in range(1):
        start = time.time()
        batch_x, _ = videos_frames.__getitem__(i)
        end = time.time()
        print('Time taken to load a single batch of {} videos with 16 frames each: {}'.format(
            8, end - start))
        print(batch_x.shape)