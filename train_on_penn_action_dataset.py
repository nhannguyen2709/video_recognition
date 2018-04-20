import argparse
import os

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from dataloader.keras_data import PennAction
from keras_models import VGG19_SpatialMotionTemporalGRU, MultiGPUModel

parser = argparse.ArgumentParser(
    description='Training the spatial motion temporal network')
parser.add_argument(
    '--epochs',
    default=100,
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
    default=16,
    type=int,
    metavar='N',
    help='number of frames sampled from a single video')
parser.add_argument(
    '--num-classes',
    default=15,
    type=int,
    metavar='N',
    help='number of action classes')
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
    '--num-gpus',
    default=2,
    type=int,
    metavar='N',
    help='number of GPUs on the device')


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos_frames = PennAction(
        frames_path='data/Penn_Action/train/frames',
        labels_path='data/Penn_Action/train/labels',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes)

    valid_videos_frames = PennAction(
        frames_path='data/Penn_Action/validation/frames',
        labels_path='data/Penn_Action/validation/labels',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes,
        shuffle=False)

    model = VGG19_SpatialMotionTemporalGRU(
        frames_input_shape=(
            args.num_frames_sampled,
            224,
            224,
            3),
        poses_input_shape=(
            args.num_frames_sampled,
            26),
        classes=args.num_classes,
        finetune_conv_layers=False)

    # if os.path.exists('checkpoint/spatial_temporal/weights.best.hdf5'):
        # model.load_weights('checkpoint/spatial_temporal/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.train_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, verbose=1)
    save_best = ModelCheckpoint(
        'checkpoint/spatial_temporal/weights.best.hdf5',
        monitor='eval_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    
    callbacks = [save_best, reduce_lr]

    # # multi-gpu training
    # parallel_model = MultiGPUModel(model, gpus=args.num_gpus)
    # parallel_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])
    # parallel_model.fit_generator(
    #     generator=train_videos_frames,
    #     epochs=args.epochs,
    #     callbacks=callbacks,
    #     workers=args.num_workers,
    #     validation_data=valid_videos_frames,
    #     initial_epoch=args.initial_epoch)

    # single-gpu training    
    model.fit_generator(
        generator=train_videos_frames,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=args.num_workers,
        validation_data=valid_videos_frames,
        initial_epoch=args.initial_epoch)


def train_with_finetune():
    global args
    args = parser.parse_args()

    train_videos_frames = PennAction(
        frames_path='data/Penn_Action/train/frames',
        labels_path='data/Penn_Action/train/labels',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes)

    valid_videos_frames = PennAction(
        frames_path='data/Penn_Action/validation/frames',
        labels_path='data/Penn_Action/validation/labels',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        num_classes=args.num_classes,
        shuffle=False)

    model = VGG19_SpatialMotionTemporalGRU(
        frames_input_shape=(
            args.num_frames_sampled,
            224,
            224,
            3),
        poses_input_shape=(
            args.num_frames_sampled,
            26),
        classes=args.num_classes,
        finetune_conv_layers=True)

    # if os.path.exists('checkpoint/spatial_temporal/weights.best.hdf5'):
        # model.load_weights('checkpoint/spatial_temporal/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.train_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, verbose=1)
    save_best = ModelCheckpoint(
        'checkpoint/spatial_temporal/weights.best.hdf5',
        monitor='eval_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    
    callbacks = [save_best, reduce_lr]

    # # multi-gpu training
    # parallel_model = MultiGPUModel(model, gpus=args.num_gpus)
    # parallel_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])
    # parallel_model.fit_generator(
    #     generator=train_videos_frames,
    #     epochs=args.epochs,
    #     callbacks=callbacks,
    #     workers=args.num_workers,
    #     validation_data=valid_videos_frames,
    #     initial_epoch=args.initial_epoch)

    # single-gpu training    
    model.fit_generator(
        generator=train_videos_frames,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=args.num_workers,
        validation_data=valid_videos_frames,
        initial_epoch=args.initial_epoch)


if __name__ == '__main__':
    train()
    K.clear_session()
    train_with_finetune()
