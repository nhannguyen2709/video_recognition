import argparse, os, pickle, time
import cv2
import numpy as np
from keras_models import VGG19_SpatialTemporalGRU

parser = argparse.ArgumentParser(description='Training the motion temporal network')
parser.add_argument('--weights-path', default='checkpoint/spatial_temporal/weights.best.hdf5',
    type=str, metavar='PATH', help='path to best model weights')
parser.add_argument('--input-loc', default='data/test_videos/C0126.MP4',
    type=str, metavar='PATH', help='input video')
parser.add_argument('--output-loc', default='outputs/action_no_pose.avi',
    type=str, metavar='PATH', help='output video')
parser.add_argument('--num-frames', default=256,
    type=int, metavar='N', help='number of frames used to recognize action')

if __name__=='__main__':
    global args
    args = parser.parse_args()
    
    labels = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'PickUpObject', 'TakeOffJacket', 'TakeOffShoes']
    model = VGG19_SpatialTemporalGRU(frames_input_shape=(args.num_frames, 224, 224, 3), classes=7)
    model.summary()
    model.load_weights(args.weights_path)

    videos_frames = np.zeros((args.num_frames, 224, 224, 3))
    output_video_size = (1920, 1080)

    time_start = time.time()
    cap = cv2.VideoCapture(os.path.join(args.input_loc, ))
    true_action = args.input_loc.split('/')[-1]
    out = cv2.VideoWriter(args.output_loc, cv2.VideoWriter_fourcc('M','J','P','G'), 10, output_video_size)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  
    count = 0
    while(cap.isOpened()):
        _, frame = cap.read()
        
        resized_frame = cv2.resize(frame, (224, 224))
        if count < args.num_frames:
            videos_frames[count, :, :, :] = resized_frame
            prev_label = "True action: {}".format('TakeOffShoes')
            cv2.putText(frame, prev_label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        count = count + 1
        if count == args.num_frames - 1:
            softmax_pred = model.predict(videos_frames, batch_size=50, verbose=1)
            pred = np.argmax(softmax_pred)
        if count > args.num_frames:
            after_label = "Predict: {}".format(labels[pred])
            cv2.putText(frame, prev_label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, after_label, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 10), 2, cv2.LINE_AA)
        
        out.write(frame)

        if (count > (video_length-1)):
            time_end = time.time()
            print('Time taken to process video: {}'.format(time_end - time_start))
            cap.release()
            out.release()