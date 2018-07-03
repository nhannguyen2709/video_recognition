import argparse
import numpy as np
import os
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import SGD

from dataloader.keras_data import UCF101Frames
from keras_models import TSNs_SpatialStream

parser = argparse.ArgumentParser(description="Training the TSNs' spatial stream on UCF101 data")
parser.add_argument('--arch', type=str, default="Xception")
parser.add_argument('--num-segments', type=int, default=3)
parser.add_argument('--input-shape', nargs='+', type=int)
parser.add_argument('--consensus-type', '--cons', type=str, default='avg',
                    choices=['avg', 'max', 'attention'])

parser.add_argument('--train-path', type=str, metavar='PATH',
                    default='data/UCF101/train/frames/')
parser.add_argument('--val-path', type=str, metavar='PATH',
                    default='data/UCF101/validation/frames/')
parser.add_argument('--filepath', type=str, metavar='PATH',
                    default='checkpoint/ucf101_spatial_stream.hdf5',
                    help="path to checkpoint best model's state and weights")

parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='number of videos in a single mini-batch')
parser.add_argument('--train-lr', default=0.001, type=float, metavar='LR',
                    help='learning rate initialized')
parser.add_argument('--num-workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')


def schedule(epoch, lr):
    if epoch + 1 % 8 == 0:
        return lr * 0.1
    else:
        return lr


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos = UCF101Frames(
        frames_path=args.train_path,
        batch_size=args.batch_size,
        input_shape=tuple(args.input_shape),
        num_segments=args.num_segments)
    valid_videos = UCF101Frames(
        frames_path=args.val_path,
        batch_size=args.batch_size,
        input_shape=tuple(args.input_shape),
        num_segments=args.num_segments,
        shuffle=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1), min_lr=1e-6,
                                  patience=2, verbose=1)
    lr_scheduler = LearningRateScheduler(schedule=schedule)
    save_best = ModelCheckpoint(
        args.filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks = [save_best, reduce_lr, lr_scheduler]

    if os.path.exists(args.filepath):
        model = load_model(args.filepath)
    else:
        model = TSNs_SpatialStream(
            input_shape=args.input_shape,
            classes=len(train_videos.labels),
            num_segments=args.num_segments,
            base_model=args.arch,
            consensus_type=args.consensus_type)
        model.compile(optimizer=SGD(lr=args.train_lr, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
    model.fit_generator(
        generator=train_videos,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=args.num_workers,
        validation_data=valid_videos)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    train()
