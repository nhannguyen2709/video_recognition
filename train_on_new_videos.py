import argparse
import os

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from dataloader.keras_data import VideosFrames
from keras_models import VGG19_SpatialTemporalGRU
from keras_callbacks import VideoLevelEvaluation

parser = argparse.ArgumentParser(
    description='Training the spatial temporal network')
parser.add_argument(
    '--epochs',
    default=500,
    type=int,
    metavar='N',
    help='number of total epochs')
parser.add_argument(
    '--batch-size',
    default=8,
    type=int,
    metavar='N',
    help='mini-batch size')
parser.add_argument(
    '--num-frames-sampled',
    default=32,
    type=int,
    metavar='N',
    help='number of frames sampled from a single video')
parser.add_argument(
    '--num-classes',
    default=101,
    type=int,
    metavar='N',
    help='number of classes')
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
    default='checkpoint/spatial_temporal/weights.{epoch:02d}-{val_acc:.4f}.hdf5',
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
parser.add_argument(
    '--interval',
    default=1,
    type=int,
    metavar='N',
    help='number of epochs before evaluating')
parser.add_argument(
    '--num-videos-eval',
    default=64,
    type=int,
    metavar='N',
    help='number of videos evaluated')


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos_frames = VideosFrames(
        data_path='data/train_videos_01',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes)

    valid_videos_frames = VideosFrames(
        data_path='data/test_videos_01',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes,
        shuffle=False)

    model = VGG19_SpatialTemporalGRU(
        frames_input_shape=(
            args.num_frames_sampled,
            224,
            224,
            3),
        classes=args.num_classes,
        finetune_conv_layers=False)

    if os.path.exists('checkpoint/spatial_temporal/weights.best.hdf5'):
        model.load_weights('checkpoint/spatial_temporal/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.train_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='eval_acc', verbose=1,
                                 mode='max', period=20)
    reduce_lr = ReduceLROnPlateau(monitor='eval_loss', factor=0.2,
                                  patience=10, verbose=1)
    save_best = ModelCheckpoint(
        'checkpoint/spatial_temporal/weights.best.hdf5',
        monitor='eval_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    video_level_eval = VideoLevelEvaluation(
        validation_data=valid_videos_frames,
        interval=args.interval,
        num_videos_eval=args.num_videos_eval,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes,
        num_segments=10)
    callbacks = [video_level_eval, checkpoint, save_best, reduce_lr]

    model.fit_generator(
        generator=train_videos_frames,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=args.num_workers,
        initial_epoch=args.initial_epoch)


def train_with_finetune():
    train_videos_frames = VideosFrames(
        data_path='data/train_videos_01',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes)

    valid_videos_frames = VideosFrames(
        data_path='data/test_videos_01',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes,
        shuffle=False)

    model = VGG19_SpatialTemporalGRU(
        frames_input_shape=(
            args.num_frames_sampled,
            224,
            224,
            3),
        classes=args.num_classes,
        finetune_conv_layers=True)

    if os.path.exists('checkpoint/spatial_temporal/weights.best.hdf5'):
        model.load_weights('checkpoint/spatial_temporal/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.finetune_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='eval_acc', verbose=1,
                                 mode='max', period=20)
    reduce_lr = ReduceLROnPlateau(monitor='eval_loss', factor=0.2,
                                  patience=10, verbose=1)
    save_best = ModelCheckpoint(
        'checkpoint/spatial_temporal/weights.best.hdf5',
        monitor='eval_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    video_level_eval = VideoLevelEvaluation(
        validation_data=valid_videos_frames,
        interval=args.interval,
        num_videos_eval=args.num_videos_eval,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes,
        num_segments=20)
    callbacks = [video_level_eval, checkpoint, save_best, reduce_lr]

    model.fit_generator(
        generator=train_videos_frames,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=args.num_workers,
        initial_epoch=args.initial_epoch)


if __name__ == '__main__':
    train()
    K.clear_session()
    train_with_finetune()
