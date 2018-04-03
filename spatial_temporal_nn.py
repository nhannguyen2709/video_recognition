import argparse, os

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from dataloader.keras_data import VideosFrames, VideosPoses
from keras_model import VGG19_SpatialTemporalGRU, MotionTemporalGRU

# limit tensorflow's memory usage
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Training the spatial temporal network')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--num-frames-sampled', default=32, type=int, metavar='N', help='number of frames sampled from a single video')
parser.add_argument('--train-learning-rate', default=1e-3, type=float, metavar='LR', help='learning rate of train stage')
parser.add_argument('--finetune-learning-rate', default=1e-5, type=float, metavar='LR', help='learning rate of finetune stage')
parser.add_argument('--checkpoint-path', default='checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--num-workers', default=4, type=int, metavar='N', help='maximum number of processes to spin up')
parser.add_argument('--initial-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def train():
    global args
    args = parser.parse_args()
    print(args)

    train_videos_frames = VideosFrames(data_path='data/NewVideos/train_videos/',
                                       frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                       batch_size=args.batch_size, num_frames_sampled=args.num_frames_sampled)
    
    validation_videos_frames = VideosFrames(data_path='data/NewVideos/validation_videos',
                                            frame_counts_path='dataloader/dic/merged_frame_count.pickle',
                                            batch_size=args.batch_size, num_frames_sampled=args.num_frames_sampled)

    model = VGG19_SpatialTemporalGRU(frames_input_shape=(args.num_frames_sampled, 224, 224, 3), 
                                     classes=7, finetune_conv_layers=False)

    if os.path.exists('checkpoint/weights.best.hdf5'):
        model.load_weights('checkpoint/weights.best.hdf5')
    model.summary()
    model.compile(optimizer=Adam(lr=args.train_learning_rate), 
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc', verbose=1, 
                                 mode='max', period=5)
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    save_best = ModelCheckpoint('checkpoint/weights.best.hdf5',
                                monitor='val_acc', verbose=1, 
                                save_best_only=True, mode='max')
    callbacks = [checkpoint, save_best, early_stopping, reduce_lr]

    model.fit_generator(generator=train_videos_frames, epochs=args.epochs, 
                        callbacks=callbacks, validation_data=validation_videos_frames,
                        workers=args.num_workers, initial_epoch=args.initial_epoch)


# def evaluate_1video():
#     global args
#     args = parser.parse_args()
#     print(args)

#     video_path = 'data/NewVideos/train_videos_multiple_actions/v_MultipleActions_g01_c01.MP4'
#     single_video_frames = np.array([resize(imread(os.path.join(video_path, frame)), (224, 224)) 
#                                     for frame in sorted(os.listdir(video_path))[:args.num_frames_used] if frame.endswith('.jpg')])
#     single_video_poses = np.load(os.path.join(video_path, 'poses.npy'))
#     frame_counts = single_video_poses.shape[0]
#     single_video_poses = single_video_poses.reshape(frame_counts, -1)
#     single_video_poses = single_video_poses[:args.num_frames_used]
#     single_video_poses[np.isnan(single_video_poses)] = -1. # fill missing coordinates with -1
    
#     with tf.device('/gpu:0'):
#         model = TemporalGRU(frames_features_input_shape=(args.num_frames_used, 224, 224, 3), 
#                             poses_input_shape=(args.num_frames_used, 54),
#                             classes=7)
#     if os.path.exists('checkpoint/weights.best.hdf5'):
#         model.load_weights('checkpoint/weights.best.hdf5')
#     model.predict(x=[single_video_frames, single_video_poses])


if __name__=='__main__':
    train()
