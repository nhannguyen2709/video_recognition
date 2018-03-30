import os
import numpy as np
from cv2 import imread, resize
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from model import VideoSequence, TemporalGRU

# limit tensorflow's memory usage
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Training and evaluating model')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--num-frames-used', default=250, type=int, metavar='N', help='number of frames used')
parser.add_argument('--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--checkpoint-path', default='checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--num-workers', default=4, type=int, metavar='N', help='maximum number of processes to spin up')
parser.add_argument('--multiprocessing', default=True, type=bool, help='if True, use process-based threading')
parser.add_argument('--initial-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def train():
    global args
    args = parser.parse_args()
    print(args)

    video_sequence = VideoSequence(data_dir='data/NewVideos/videos_frames/',
                                   frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                   batch_size=args.batch_size, num_frames_used=args.num_frames_used)

    with K.device('gpu0'):
        model = TemporalGRU(frames_features_input_shape=(args.num_frames_used, 512), 
                            poses_input_shape=(args.num_frames_used, 54),
                            classes=7)

    if os.path.exists('checkpoint/weights.best.hdf5'):
        model.load_weights('checkpoint/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.learning_rate), 
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc', verbose=1, 
                                 mode='max', period=10)
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=7)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    save_best = ModelCheckpoint('checkpoint/weights.best.hdf5',
                                monitor='val_acc', verbose=1, 
                                save_best_only=True, mode='max')
    callbacks = [checkpoint, save_best, early_stopping, reduce_lr]

    model.fit_generator(video_sequence, epochs=args.epochs, 
                        callbacks=callbacks, workers=args.num_workers, 
                        use_multiprocessing=args.multiprocessing, initial_epoch=args.initial_epoch)


def evaluate_1video():
    global args
    args = parser.parse_args()
    print(args)

    video_path = 'data/NewVideos/train_videos_multiple_actions/v_MultipleActions_g01_c01.MP4'
    single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224)) 
                                    for frame in sorted(os.listdir(video_path))[:args.num_frames_used] if frame.endswith('.jpg')])
    single_video_poses = np.load(os.path.join(video_path, 'poses.npy'))
    frame_counts = single_video_poses.shape[0]
    single_video_poses = single_video_poses.reshape(frame_counts, -1)
    single_video_poses = single_video_poses[:args.num_frames_used]
    single_video_poses[np.isnan(single_video_poses)] = -1. # fill missing coordinates with -1
    
    with K.device('gpu0'):
        model = TemporalGRU(frames_features_input_shape=(args.num_frames_used, 224, 224, 3), 
                            poses_input_shape=(args.num_frames_used, 54),
                            classes=7)
    if os.path.exists('checkpoint/weights.best.hdf5'):
        model.load_weights('checkpoint/weights.best.hdf5')
    model.predict(x=[single_video_frames, single_video_poses])


if __name__=='__main__':
    train()
