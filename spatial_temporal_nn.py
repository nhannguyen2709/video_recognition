import argparse
import os
import pickle

import numpy as np
from cv2 import imread, resize

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.optimizers import Adam
from keras.backend import tensorflow_backend as K
from keras_models import VGG19_SpatialTemporalGRU
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import resample

from dataloader.keras_data import VideosFrames
# limit tensorflow's memory usage
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# K.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(
    description='Training the spatial temporal network')
parser.add_argument('--epochs', default=500,
                    type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--num-frames-sampled', default=32,
                    type=int, metavar='N', help='number of frames sampled from a single video')
parser.add_argument('--train-lr', default=1e-3,
                    type=float, metavar='LR', help='learning rate of train stage')
parser.add_argument('--finetune-lr', default=1e-5,
                    type=float, metavar='LR', help='learning rate of finetune stage')
parser.add_argument('--checkpoint-path', default='checkpoint/spatial_temporal/weights.{epoch:02d}-{val_acc:.4f}.hdf5',
                    type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--num-workers', default=4,
                    type=int, metavar='N', help='maximum number of processes to spin up')
parser.add_argument('--initial-epoch', default=0,
                    type=int, metavar='N', help='manual epoch number (useful on restarts)')


class VideoLevelEvaluation(Callback):
    def __init__(self, data_path, frame_counts_path,
                 validation_data, interval=5, num_videos_eval=4,
                 num_frames_sampled=32, num_segments=10, num_classes=7):
        super(Callback, self).__init__()

        frame_counts = pickle.load(open(frame_counts_path, 'rb'))
        video_filenames = validation_data.video_filenames
        self.video_paths = validation_data.x
        self.y_val = np.array(validation_data.y)
        self.valid_video_frame_counts = [frame_counts[filename]
                                         for filename in video_filenames]
        self.interval = interval
        self.num_videos_eval = num_videos_eval
        self.num_frames_sampled = num_frames_sampled
        self.num_segments = num_segments
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # sample some videos to evaluate from the validation set
            eval_video_paths, eval_y_val, eval_video_frame_counts = resample(self.video_paths, self.y_val, self.valid_video_frame_counts,
                                                                             replace=False, n_samples=self.num_videos_eval)
            avg_eval_video_preds = np.zeros(
                shape=(self.num_videos_eval, self.num_classes))

            for _ in range(self.num_segments):
                eval_video_frames = []
                for video_path, frame_counts in zip(eval_video_paths, eval_video_frame_counts):
                    frames_snippet = self.sample_frames_snippet(
                        video_path, frame_counts)
                    single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224))
                                                    for frame in frames_snippet])
                    eval_video_frames.append(single_video_frames)
                eval_video_frames = np.array(eval_video_frames)
                eval_video_preds = self.model.predict(
                    eval_video_frames, verbose=0)
                # average the probabilities across segments
                avg_eval_video_preds += eval_video_preds/self.num_segments

            eval_loss = log_loss(y_true=to_categorical(eval_y_val, num_classes=self.num_classes),
                                 y_pred=avg_eval_video_preds)
            eval_acc = accuracy_score(y_true=eval_y_val,
                                      y_pred=np.argmax(avg_eval_video_preds, axis=1))
            print("\n Epoch: {:d} - eval. loss: {:.4f} - eval. acc.: {:.4f}".format(
                epoch+1, eval_loss, eval_acc))

    def sample_frames_snippet(self, video_path, frame_counts):
        frames = np.array([filename for filename in sorted(
            os.listdir(video_path)) if filename.endswith('.jpg')])
        start_frameidx_in_snippet = np.random.randint(
            0, frame_counts - self.num_frames_sampled)
        end_frameidx__in_snippet = start_frameidx_in_snippet + self.num_frames_sampled
        framesidx_snippet = range(
            start_frameidx_in_snippet, end_frameidx__in_snippet)
        return frames[framesidx_snippet]


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos_frames = VideosFrames(data_path='data/NewVideos/train_videos/',
                                       frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                       batch_size=args.batch_size, num_frames_sampled=args.num_frames_sampled)

    valid_videos_frames = VideosFrames(data_path='data/NewVideos/validation_videos',
                                       frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                       batch_size=args.batch_size, num_frames_sampled=args.num_frames_sampled,
                                       shuffle=False)

    model = VGG19_SpatialTemporalGRU(frames_input_shape=(args.num_frames_sampled, 224, 224, 3),
                                     classes=7, finetune_conv_layers=False)

    if os.path.exists('checkpoint/spatial_temporal/weights.best.hdf5'):
        model.load_weights('checkpoint/spatial_temporal/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.train_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc', verbose=1,
                                 mode='max', period=15)
    # early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=50)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=15, min_lr=0.001)
    save_best = ModelCheckpoint('checkpoint/spatial_temporal/weights.best.hdf5',
                                monitor='train_loss', verbose=1,
                                save_best_only=True, mode='max')
    video_level_eval = VideoLevelEvaluation(validation_data=valid_videos_frames,
                                            data_path='data/NewVideos/validation_videos',
                                            frame_counts_path='dataloader/dic/merged_frame_count.pickle')
    callbacks = [checkpoint, reduce_lr, save_best, video_level_eval]

    model.fit_generator(generator=train_videos_frames, epochs=args.epochs,
                        callbacks=callbacks, validation_data=valid_videos_frames,
                        workers=args.num_workers, initial_epoch=args.initial_epoch)


def train_with_finetune():
    train_videos_frames = VideosFrames(data_path='data/NewVideos/train_videos/',
                                       frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                       batch_size=args.batch_size, num_frames_sampled=args.num_frames_sampled)

    valid_videos_frames = VideosFrames(data_path='data/NewVideos/validation_videos',
                                       frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                       batch_size=args.batch_size, num_frames_sampled=args.num_frames_sampled,
                                       shuffle=False)

    model = VGG19_SpatialTemporalGRU(frames_input_shape=(args.num_frames_sampled, 224, 224, 3),
                                     classes=7, finetune_conv_layers=True)

    if os.path.exists('checkpoint/spatial_temporal/weights.best.hdf5'):
        model.load_weights('checkpoint/spatial_temporal/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.finetune_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc', verbose=1,
                                 mode='max', period=15)
    # early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=50)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=15, min_lr=0.001)
    save_best = ModelCheckpoint('checkpoint/spatial_temporal/weights.best.hdf5',
                                monitor='train_loss', verbose=1,
                                save_best_only=True, mode='max')
    video_level_eval = VideoLevelEvaluation(validation_data=valid_videos_frames,
                                            data_path='data/NewVideos/validation_videos',
                                            frame_counts_path='dataloader/dic/merged_frame_count.pickle')
    callbacks = [checkpoint, reduce_lr, save_best, video_level_eval]

    model.fit_generator(generator=train_videos_frames, epochs=args.epochs,
                        callbacks=callbacks, validation_data=valid_videos_frames,
                        workers=args.num_workers, initial_epoch=args.initial_epoch)


if __name__ == '__main__':
    train()
    K.clear_session()
    train_with_finetune()