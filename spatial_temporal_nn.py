import argparse, os

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from dataloader.keras_data import VideosFrames
from keras_models import VGG19_SpatialTemporalGRU
from keras_callbacks import VideoLevelEvaluation

# limit tensorflow's memory usage
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# K.set_session(tf.Session(config=config))

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
    help='mini-batch size (default: 16)')
parser.add_argument(
    '--num-frames-sampled',
    default=32,
    type=int,
    metavar='N',
    help='number of frames sampled from a single video')
parser.add_argument(
    '--num-frames-skipped',
    default=2,
    type=int,
    metavar='N',
    help='number of frames skipped when sampling')
parser.add_argument(
    '--num-classes',
    default=104,
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


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos_frames = VideosFrames(
        data_path='data/train_videos_02',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_frames_skipped=args.num_frames_skipped,
        num_classes=args.num_classes)

    valid_videos_frames = VideosFrames(
        data_path='data/test_videos_02',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_frames_skipped=args.num_frames_skipped,
        num_classes=args.num_classes,
        shuffle=False)

    model = VGG19_SpatialTemporalGRU(
        frames_input_shape=(
            int(args.num_frames_sampled / args.num_frames_skipped),
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
        data_path='data/test_videos_02',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        num_segments=10)
    callbacks = [checkpoint, reduce_lr, save_best, video_level_eval]

    model.fit_generator(
        generator=train_videos_frames,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=valid_videos_frames,
        workers=args.num_workers,
        initial_epoch=args.initial_epoch)


def train_with_finetune():
    train_videos_frames = VideosFrames(
        data_path='data/train_videos_02',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_frames_skipped=args.num_frames_skipped,
        num_classes=args.num_classes)

    valid_videos_frames = VideosFrames(
        data_path='data/test_videos_02',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_frames_skipped=args.num_frames_skipped,
        num_classes=args.num_classes,
        shuffle=False)

    model = VGG19_SpatialTemporalGRU(
        frames_input_shape=(
            int(args.num_frames_sampled / args.num_frames_skipped),
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
        data_path='data/test_videos_02',
        frame_counts_path='dataloader/dic/merged_frame_count.pickle',
        num_segments=10)
    callbacks = [checkpoint, reduce_lr, save_best, video_level_eval]

    model.fit_generator(
        generator=train_videos_frames,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=valid_videos_frames,
        workers=args.num_workers,
        initial_epoch=args.initial_epoch)


if __name__ == '__main__':
    train()
    K.clear_session()
    train_with_finetune()
