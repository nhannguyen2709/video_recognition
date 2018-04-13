import os
import pickle
import time
import gc

from cv2 import imread, resize
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import resample

from keras.callbacks import Callback
from keras.utils import to_categorical


class VideoLevelEvaluation(Callback):
    def __init__(self, validation_data, interval, num_videos_eval,
                 num_frames_sampled, num_segments, num_classes):
        super(Callback, self).__init__()

        self.data_path = validation_data.data_path
        self.frame_counts = validation_data.frame_counts
        self.videos, self.videos_dict = validation_data.x, validation_data.videos_dict
        self.y_val = validation_data.y
        self.interval = interval
        self.num_videos_eval = num_videos_eval
        self.num_frames_sampled = num_frames_sampled
        self.num_segments = num_segments
        self.num_classes = num_classes

    def sample_clips(self, video):
        clips = np.array(self.videos_dict[video]['clips'])
        num_clips = self.videos_dict[video]['num_clips']
        sampled_clips_idx = sorted(np.random.choice(
            num_clips, size=4, replace=False))
        return clips[sampled_clips_idx]

    def sample_frames_snippet(self, clip_path, frame_count):
        frames = np.array([filename for filename in sorted(
            os.listdir(clip_path)) if filename.endswith('.jpg')])
        frames_snippet_idx = sorted(np.random.choice(
            frame_count, size=int(self.num_frames_sampled / 4), replace=False))
        return frames[frames_snippet_idx]

    def get_frame_count(self, sampled_clips):
        sampled_clip_frame_counts = [self.frame_counts[clip]
                                     for clip in sampled_clips]
        return sampled_clip_frame_counts

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % self.interval == 0:
            start = time.time()
            # sample some videos to evaluate from the validation set
            eval_videos, eval_y_val = resample(
                self.videos, self.y_val, replace=False, n_samples=self.num_videos_eval)
            eval_videos_pred = np.zeros(
                shape=(self.num_videos_eval, self.num_classes))
            avg_eval_videos_pred = np.zeros(
                shape=(self.num_videos_eval, self.num_classes))

            for _ in range(self.num_segments):
                for i, video  in enumerate(eval_videos):
                    sampled_clips = self.sample_clips(video)
                    clip_paths = [os.path.join(self.data_path, clip)
                                  for clip in sampled_clips]
                    clip_frame_counts = self.get_frame_count(sampled_clips)
                    clips_frames = [
                        self.sample_frames_snippet(
                            clip_path, frame_count) for clip_path, frame_count in zip(
                            clip_paths, clip_frame_counts)]
                    video_frames = []
                    for clip_frames, clip_path in zip(
                            clips_frames, clip_paths):
                        video_frames.append([resize(imread(os.path.join(clip_path, frame)), (224, 224))
                                             for frame in clip_frames])
                    video_frames = np.reshape(
                        video_frames, (self.num_frames_sampled, 224, 224, 3))
                    video_frames = np.expand_dims(video_frames, axis=0)
                    video_frames = video_frames / 255.
                    eval_videos_pred[i] = self.model.predict(
                        video_frames, verbose=0)
                    del video_frames
                    gc.collect()
                # average the probabilities across segments
                avg_eval_videos_pred += eval_videos_pred / self.num_segments

            eval_loss = log_loss(
                y_true=to_categorical(
                    eval_y_val,
                    num_classes=self.num_classes),
                y_pred=avg_eval_videos_pred)
            eval_acc = accuracy_score(
                y_true=eval_y_val, y_pred=np.argmax(
                    avg_eval_videos_pred, axis=1))
            logs['eval_loss'] = eval_loss
            logs['eval_acc'] = eval_acc
            end = time.time()
            print(
                "- evaluation time: {:.2f}s - eval_loss: {:.4f} - eval_acc.: {:.4f}\n".format(
                    end -
                    start,
                    eval_loss,
                    eval_acc))
