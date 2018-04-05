import time
import cv2
import os
import pickle 
import numpy as np
from models import VGG19_frame_features_extractor, spatial_rnn

weights_dir = 'weights/'
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)
checkpoint_path = os.path.join(weights_dir, 'weights.best.hdf5')

classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'PickUpObject', 'TakeOffJacket', 'TakeOffShoes']
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

vgg19_frame_features_extractor = VGG19_frame_features_extractor()
spatial_rnn = spatial_rnn()
spatial_rnn.load_weights(os.path.join(checkpoint_path))

tunable_num_frames = 150
input_loc = 'data/NewVideos/test'
concatenated_frames = np.zeros((tunable_num_frames, 224, 224, 3))

output_video_size = (1920, 1080)
# Log the time  
time_start = time.time()
# Start capturing the feed
cap = cv2.VideoCapture(os.path.join(input_loc, 'C0126.MP4'))
true_action = input_loc.split('/')[-1]
out = cv2.VideoWriter('outpy1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, output_video_size)
# Find the number of frames
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
count = 0
# Start converting the video
while(cap.isOpened()):
    # Extract the frame
    _, frame = cap.read()
    
    # Resize and pass to our network
    resized_frame = cv2.resize(frame, (224, 224))
    if count < tunable_num_frames:
        concatenated_frames[count, :, :, :] = resized_frame
        prev_label = "True action: {}".format('TakeOffShoes')
        cv2.putText(frame, prev_label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    count = count + 1
    if count == tunable_num_frames - 1:
        video_features = vgg19_frame_features_extractor.predict(concatenated_frames, batch_size=50, verbose=1)
        video_features = np.expand_dims(video_features, axis=0)
        pred = np.argmax(spatial_rnn.predict(video_features))
    if count > tunable_num_frames:
        after_label = "Predict: {}".format(classes[pred])
        cv2.putText(frame, prev_label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, after_label, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 10), 2, cv2.LINE_AA)
    
    out.write(frame)

    # If there are no more frames left
    if (count > (video_length-1)):
        # Log the time again
        time_end = time.time()
        print('Time taken to process video: {}'.format(time_end - time_start))
        # Release the feed
        cap.release()
        out.release()