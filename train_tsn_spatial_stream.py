import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import SGD

from dataloader.keras_data import UCF101Frames
from keras_models import MultiGPUModel, TemporalSegmentNetworks_SpatialStream

parser = argparse.ArgumentParser(
    description='Training the Temporal Segment Networks on UCF101 data')
parser.add_argument(
    '--filepath',
    default='checkpoint/ucf101_spatial_stream.hdf5',
    type=str,
    metavar='PATH',
    help="path to checkpoint best model's state and weights")
parser.add_argument(
    '--epochs',
    default=20,
    type=int,
    metavar='N',
    help='number of total epochs')
parser.add_argument(
    '--batch-size',
    default=8,
    type=int,
    metavar='N',
    help='number of videos in a single mini-batch')
parser.add_argument(
    '--train-lr',
    default=0.001,
    type=float,
    metavar='LR',
    help='learning rate initialized')
parser.add_argument(
    '--num-workers',
    default=12,
    type=int,
    metavar='N',
    help='maximum number of processes to spin up')


def schedule(epoch, lr):
    if epoch + 1 % 10 == 0:
        return lr * 0.1
    else:
        return lr


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos = UCF101Frames(
        frames_path='data/UCF101/train/frames/',
        batch_size=args.batch_size)
    valid_videos = UCF101Frames(
        frames_path='data/UCF101/validation/frames',
        batch_size=args.batch_size,
        shuffle=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
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
        model = TemporalSegmentNetworks_SpatialStream(
            input_shape=(299, 299, 3), dropout_prob=0.8,
            classes=len(train_videos.labels))
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
    train()
