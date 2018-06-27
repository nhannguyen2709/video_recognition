import argparse
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from optimizers import SGD
from keras.utils import multi_gpu_model

from dataloader.keras_data import UCF101Flows
from keras_models import TemporalSegmentNetworks_MotionStream

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
    default=80,
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
    default=0.0005,
    type=float,
    metavar='LR',
    help='learning rate initialized')
parser.add_argument(
    '--num-workers',
    default=12,
    type=int,
    metavar='N',
    help='maximum number of processes to spin up')


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

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                  min_lr=1e-6,
                                  patience=2, verbose=1)
    save_best = ModelCheckpoint(
        args.filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks = [save_best, reduce_lr]
    lr_multipliers = {}
    lr_multipliers['block1_conv1/kernel:0'] = 10

    if os.path.exists(args.filepath):
        model = load_model(args.filepath)
    else:        
        model = TemporalSegmentNetworks_MotionStream(
            input_shape=(299, 299, 20), dropout_prob=0.7,
            classes=len(train_videos.labels))
    
    model.compile(optimizer=SGD(lr=args.train_lr, momentum=0.9, multipliers=lr_multipliers),
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
