import os, pickle, time

from cv2 import imread, resize
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import resample

from keras.callbacks import Callback
from keras.utils import to_categorical


class VideoLevelEvaluation(Callback):
    def __init__(self, data_path, frame_counts_path,
                 validation_data, interval=1, num_videos_eval=4,
                 num_frames_sampled=32, num_segments=10, num_classes=7):
        super(Callback, self).__init__()

        frame_counts = pickle.load(open(frame_counts_path, 'rb'))
        video_filenames = validation_data.video_filenames
        self.video_paths = validation_data.x
        self.y_val = validation_data.y
        self.valid_video_frame_counts = [frame_counts[filename]
                                         for filename in video_filenames]
        self.interval = interval
        self.num_videos_eval = num_videos_eval
        self.num_frames_sampled = num_frames_sampled
        self.num_segments = num_segments
        self.num_classes = num_classes

    def sample_frames_snippet(self, video_path, frame_counts):
        frames = np.array([filename for filename in sorted(
            os.listdir(video_path)) if filename.endswith('.jpg')])
        start_frameidx_in_snippet = np.random.randint(
            0, frame_counts - self.num_frames_sampled)
        end_frameidx__in_snippet = start_frameidx_in_snippet + self.num_frames_sampled
        framesidx_snippet = range(
            start_frameidx_in_snippet, end_frameidx__in_snippet)
        return frames[framesidx_snippet]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % self.interval == 0:
            start = time.time()
            # sample some videos to evaluate from the validation set
            eval_video_paths, eval_y_val, eval_video_frame_counts = resample(
                self.video_paths, self.y_val, self.valid_video_frame_counts, replace=False, n_samples=self.num_videos_eval)
            avg_eval_video_preds = np.zeros(
                shape=(self.num_videos_eval, self.num_classes))

            for _ in range(self.num_segments):
                eval_video_frames = []
                for video_path, frame_counts in zip(
                        eval_video_paths, eval_video_frame_counts):
                    frames_snippet = self.sample_frames_snippet(
                        video_path, frame_counts)
                    single_video_frames = np.array([resize(imread(os.path.join(
                        video_path, frame)), (224, 224)) for frame in frames_snippet])
                    eval_video_frames.append(single_video_frames)
                eval_video_frames = np.array(eval_video_frames)
                eval_video_preds = self.model.predict(
                    eval_video_frames, verbose=0)
                # average the probabilities across segments
                avg_eval_video_preds += eval_video_preds / self.num_segments

            eval_loss = log_loss(
                y_true=to_categorical(
                    eval_y_val,
                    num_classes=self.num_classes),
                y_pred=avg_eval_video_preds)
            eval_acc = accuracy_score(
                y_true=eval_y_val, y_pred=np.argmax(
                    avg_eval_video_preds, axis=1))
            logs['eval_loss'] = eval_loss
            logs['eval_acc'] = eval_acc
            end = time.time()
            print(
                "\nEpoch {:d}\n - evaluation time: {:.2f}s - eval_loss: {:.4f} - eval_acc.: {:.4f}".format(
                    epoch + 1, end - start, eval_loss, eval_acc))