import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam

from dataloader.keras_data import PennAction
from keras_models import VGG19_SpatialMotionTemporalGRU, MultiGPUModel

parser = argparse.ArgumentParser(
    description='Training the spatial motion temporal network')
parser.add_argument(
    '--filepath',
    default='checkpoint/penn_action.hdf5',
    type=str,
    metavar='PATH',
    help="path to checkpoint best model's state and weights")
parser.add_argument(
    '--epochs',
    default=50,
    type=int,
    metavar='N',
    help='number of total epochs')
parser.add_argument(
    '--batch-size',
    default=4,
    type=int,
    metavar='N',
    help='number of videos in a single mini-batch')
parser.add_argument(
    '--num-frames-sampled',
    default=16,
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
    default='multi',
    type=str,
    help='gpu mode (single or multi)')


def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos = PennAction(
        frames_path='data/PennAction/train/frames/',
        labels_path='data/PennAction/train/labels',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled)
    valid_videos = PennAction(
        frames_path='data/PennAction/validation/frames',
        labels_path='data/PennAction/validation/labels',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        shuffle=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                  patience=5, verbose=1)
    save_best = ModelCheckpoint(
        args.filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks = [save_best, reduce_lr]

    with tf.device('/CPU:0'):
        if os.path.exists(args.filepath):
            model = load_model(args.filepath)
        else:  # initialize the model if file path doesn't exist
            model = VGG19_SpatialMotionTemporalGRU(
                frames_input_shape=(
                    args.num_frames_sampled,
                    224,
                    224,
                    3),
                poses_input_shape=(
                    args.num_frames_sampled,
                    26),
                classes=len(train_videos.labels))

    print('Train the GRU component only')
    if args.gpu_mode == 'single':
        model.compile(optimizer=Adam(lr=args.train_lr, decay=1e-5),
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
            optimizer=model.optimizer,
            loss='categorical_crossentropy',
            metrics=['acc'])
        parallel_model.fit_generator(
            generator=train_videos,
            epochs=args.epochs,
            callbacks=callbacks,
            workers=args.num_workers,
            validation_data=valid_videos)
        return K.get_value(parallel_model.optimizer.lr)


def train_with_finetune(prev_stage_lr):
    global args
    args = parser.parse_args()

    train_videos = PennAction(
        frames_path='data/PennAction/train/frames/',
        labels_path='data/PennAction/train/labels',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled)
    valid_videos = PennAction(
        frames_path='data/PennAction/validation/frames',
        labels_path='data/PennAction/validation/labels',
        batch_size=args.batch_size,
        num_frames_sampled=args.num_frames_sampled,
        shuffle=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                  patience=5, verbose=1)
    save_best = ModelCheckpoint(
        args.filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks = [save_best, reduce_lr]

    with tf.device('/CPU:0'):
        model = load_model(args.filepath)
    model.layers[-9].trainable = True
    model.layers[-10].trainable = True

    print('Fine-tune top 2 convolutional layers of VGG19')
    if args.gpu_mode == 'single':
        model.compile(optimizer=Adam(lr=K.get_value(model.optimizer.lr) * 0.5, decay=1e-5),
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
            optimizer=Adam(lr=prev_stage_lr * 0.5, decay=1e-5),
            loss='categorical_crossentropy',
            metrics=['acc'])
        parallel_model.fit_generator(
            generator=train_videos,
            epochs=args.epochs,
            callbacks=callbacks,
            workers=args.num_workers,
            validation_data=valid_videos)


if __name__ == '__main__':
    prev_stage_lr = train()
    K.clear_session()
    train_with_finetune(prev_stage_lr)
