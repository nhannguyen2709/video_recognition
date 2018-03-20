import time
import cv2
import os

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: {}".format(video_length))
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        _, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n{} frames extracted".format(count))
            print ("It took {} seconds for conversion.".format(time_end-time_start))

if __name__=='__main__':
    for 
    video_to_frames('data/NewVideos/PickUpObject/C0090.mp4',
                    'data/NewVideos/jpegs_256/PickUpObject')