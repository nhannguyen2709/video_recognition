import argparse
import os

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.backend import tensorflow_backend as K
from dataloader.keras_data import VideosFramesPoses
from keras_models import VGG19_SpatialMotionTemporalGRU

# limit tensorflow's memory usage
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# K.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(
    description='Training the network using both frames and poses')
parser.add_argument('--epochs', default=500,
                    type=int, metavar='N', help='number of total epochs')
parser.add_argument(
    '--batch-size',
    default=4,
    type=int,
    metavar='N',
    help='mini-batch size (default: 16)')
parser.add_argument(
    '--num-frames-sampled',
    default=32,
    type=int,
    metavar='N',
    help='number of frames sampled from a single video')
parser.add_argument(
    '--train-lr',
    default=1e-3,
    type=float,
    metavar='LR',
    help='learning rate of train stage')
parser.add_argument(
    '--finetune-lr',
    default=1e-5,
    type=float,
    metavar='LR',
    help='learning rate of finetune stage')
parser.add_argument(
    '--checkpoint-path',
    default='checkpoint/frames_poses/weights.{epoch:02d}-{val_acc:.4f}.hdf5',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint')
parser.add_argument(
    '--num-workers',
    default=4,
    type=int,
    metavar='N',
    help='maximum number of processes to spin up')
parser.add_argument(
    '--initial-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos_frames = VideosFramesPoses(
        data_path='data/NewVideos/train_videos/',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled)

    validation_videos_frames = VideosFramesPoses(
        data_path='data/NewVideos/validation_videos',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled)

    model = VGG19_SpatialMotionTemporalGRU(
        frames_input_shape=(
            args.num_frames_sampled, 224, 224, 3), poses_input_shape=(
            args.num_frames_sampled, 54), classes=7, finetune_conv_layers=False)

    if os.path.exists('checkpoint/frames_poses/weights.best.hdf5'):
        model.load_weights('checkpoint/frames_poses/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.train_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc', verbose=1,
                                 mode='max', period=15)
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=50)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=15, min_lr=0.001)
    save_best = ModelCheckpoint('checkpoint/frames_poses/weights.best.hdf5',
                                monitor='val_acc', verbose=1,
                                save_best_only=True, mode='max')
    callbacks = [checkpoint, save_best, early_stopping, reduce_lr]

    model.fit_generator(
        generator=train_videos_frames,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=validation_videos_frames,
        workers=args.num_workers,
        initial_epoch=args.initial_epoch)


def train_with_finetune():
    train_videos_frames = VideosFramesPoses(
        data_path='data/NewVideos/train_videos/',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled)

    validation_videos_frames = VideosFramesPoses(
        data_path='data/NewVideos/validation_videos',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled)

    model = VGG19_SpatialMotionTemporalGRU(
        frames_input_shape=(
            args.num_frames_sampled, 224, 224, 3), poses_input_shape=(
            args.num_frames_sampled, 54), classes=7, finetune_conv_layers=True)

    if os.path.exists('checkpoint/frames_poses/weights.best.hdf5'):
        model.load_weights('checkpoint/frames_poses/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.finetune_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc', verbose=1,
                                 mode='max', period=15)
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=50)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=15, min_lr=0.001)
    save_best = ModelCheckpoint('checkpoint/frames_poses/weights.best.hdf5',
                                monitor='val_acc', verbose=1,
                                save_best_only=True, mode='max')
    callbacks = [checkpoint, save_best, early_stopping, reduce_lr]

    model.fit_generator(
        generator=train_videos_frames,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=validation_videos_frames,
        workers=args.num_workers,
        initial_epoch=args.initial_epoch)


if __name__ == '__main__':
    train()
    K.clear_session()
    train_with_finetune()
