import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import SGD

from dataloader.keras_data import UCF101Flows
from keras_models import MultiGPUModel, TemporalSegmentNetworks_MotionStream

parser = argparse.ArgumentParser(
    description='Training the Temporal Segment Networks on UCF101 data')
parser.add_argument(
    '--filepath',
    default='checkpoint/ucf101_motion_stream.hdf5',
    type=str,
    metavar='PATH',
    help="path to checkpoint best model's state and weights")
parser.add_argument(
    '--epochs',
    default=60,
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
    default=0.005,
    type=float,
    metavar='LR',
    help='learning rate initialized')
parser.add_argument(
    '--num-workers',
    default=12,
    type=int,
    metavar='N',
    help='maximum number of processes to spin up')
parser.add_argument(
    '--num-gpus',
    default=2,
    type=int,
    metavar='N',
    help='number of GPUs on the device')
parser.add_argument(
    '--gpu-mode',
    default='single',
    type=str,
    help='gpu mode (single or multi)')


def schedule(epoch, lr):
    if epoch + 1 % 36 == 0:
        return lr * 0.1
    elif epoch + 1 % 54 == 0:
        return lr * 0.1
    else:
        return lr


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos = UCF101Flows(
        frames_path='data/UCF101/train/frames/',
        batch_size=args.batch_size)
    valid_videos = UCF101Flows(
        frames_path='data/UCF101/validation/frames',
        batch_size=args.batch_size,
        shuffle=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                  patience=5, verbose=1)
    lr_scheduler = LearningRateScheduler(schedule=schedule)
    save_best = ModelCheckpoint(
        args.filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks = [save_best, reduce_lr, lr_scheduler]

    with tf.device('/CPU:0'):
        if os.path.exists(args.filepath):
            model = load_model(args.filepath)
        else:
            model = TemporalSegmentNetworks_MotionStream(
                input_shape=(299, 299, 20), dropout_prob=0.7,
                classes=len(train_videos.labels))

    if args.gpu_mode == 'single':
        model.compile(optimizer=SGD(lr=args.train_lr, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        model.fit_generator(
            generator=train_videos,
            epochs=args.epochs,
            callbacks=callbacks,
            workers=args.num_workers,
            validation_data=valid_videos)
    else:
        parallel_model = MultiGPUModel(model, gpus=args.num_gpus)
        parallel_model.compile(
            optimizer=SGD(
                lr=args.train_lr,
                momentum=0.9),
            loss='categorical_crossentropy',
            metrics=['acc'])
        parallel_model.fit_generator(
            generator=train_videos,
            epochs=args.epochs,
            callbacks=callbacks,
            workers=args.num_workers,
            validation_data=valid_videos)


if __name__ == '__main__':
    train()
