import argparse, cv2, os, time
from keras_models import VGG19_SpatialMotionTemporalGRU
import numpy as np
from scipy import io

from dataloader.keras_data import PennAction

parser = argparse.ArgumentParser(
    description='Demo on the Penn Action dataset')
parser.add_argument('--weights-path', default='checkpoint/spatial_temporal/weights.best.hdf5',
                    type=str, metavar='PATH', help='path to best model weights')
parser.add_argument('--num-frames', default=8,
    type=int, metavar='N', help='number of frames used to recognize action')


if __name__ == '__main__':
    global args
    args=parser.parse_args()
    
    penn_action = PennAction(
        frames_path='data/Penn_Action/validation/frames',
        labels_path='data/Penn_Action/validation/labels',
        batch_size=1,
        num_frames_sampled=8,
        num_classes=15,
        shuffle=False)
    labels = penn_action.labels
    num_classes = penn_action.num_classes

    model = VGG19_SpatialMotionTemporalGRU(
        frames_input_shape=(None, 224, 224, 3),
        poses_input_shape=(None, 26), 
        classes=num_classes)
    model.summary()
    model.load_weights(args.weights_path)

    input_frames = [] 
    
    frames_input_loc = 'data/Penn_Action/validation/frames/0001'
    labels_input_loc = 'data/Penn_Action/validation/labels/0001.mat'
    output_loc = 'outputs/Penn_Action_0001.avi'
    
    # Start processing the video
    time_start = time.time()
    
    video_frames = sorted(os.listdir(frames_input_loc))
    video_length = len(video_frames) 
    print('Video has {} frames'.format(video_length))

    mat = io.loadmat(labels_input_loc)
    first_frame = cv2.imread(os.path.join(frames_input_loc, video_frames[0]))
    x_max, y_max = first_frame.shape[:2]
    poses_x, poses_y = mat['x'] / x_max, mat['y'] / y_max
    poses_x_y = np.hstack([poses_x, poses_y])
    true_action = mat['action'][0]
    
    output_video_size = (x_max, y_max)
    out = cv2.VideoWriter(output_loc, cv2.VideoWriter_fourcc('M','J','P','G'), 10, output_video_size)  
    
    for i, video_frame in enumerate(video_frames):     
        frame = cv2.imread(os.path.join(frames_input_loc, video_frame))
        resized_frame = cv2.resize(frame, 
                                   (224, 224))

        prev_label = "True action: {}".format(true_action)
        cv2.putText(frame, prev_label, (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        input_frames.append(resized_frame)

        if i < args.num_frames:
            cv2.putText(frame, "Predict: ", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 10), 1, cv2.LINE_AA)            

        if i % args.num_frames == 0 and i!= 0:
            model_input_poses = np.expand_dims(poses_x_y[:i], axis=0)
            
            model_input_frames = np.array(input_frames)
            model_input_frames = model_input_frames / 255.
            model_input_frames = np.expand_dims(model_input_frames, axis=0)
            
            softmax_pred = model.predict([model_input_frames, model_input_poses], 
                                         verbose=1)
            del model_input_frames, model_input_poses
            pred = np.argmax(softmax_pred)
            after_label = "Predict: {}".format(labels[pred])

        if i > args.num_frames:
            cv2.putText(frame, after_label, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 10), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join('outputs/Penn_Action_0001', video_frame) ,frame)

        if (i > (video_length-1)):
            time_end = time.time()
            print('Time taken to process video: {}'.format(time_end - time_start))
            out.release()
