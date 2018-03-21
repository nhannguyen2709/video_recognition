import time
import cv2
import os
import pickle 

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    # Get video filename with file extension
    video_filename = input_loc.split('/')[-1]
    # Log the time  
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("\nNumber of frames: ", video_length)
    count = 0
    print("Converting video...")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        _, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/frame%#06d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames\n{} frames extracted from {}".format(count, video_filename))
            print("Took {} seconds for converting".format(round(time_end-time_start, 4)))
    return video_filename, count

if __name__=='__main__':
    # Create a frame count for videos
    frame_count = {}

    if not os.path.exists('data/NewVideos/jpegs_256'):
        os.mkdir('data/NewVideos/jpegs_256')

    for path, _, filenames in os.walk('data/NewVideos/'):
        for filename in sorted(filenames):
            path_to_video = os.path.join(path, filename)
            if path_to_video.endswith('.mp4'):
                output_loc = os.path.join('data/NewVideos/jpegs_256', filename.replace('.mp4', ''))
                if not os.path.exists(output_loc):
                    os.mkdir(output_loc)
                # Extract frames, return video filename and its frame count
                video_filename, count = video_to_frames(path_to_video, output_loc)
                frame_count[video_filename] = count
            else:
                pass

    # Dump the frame count dict into pickle file
    with open('new_videos_frame_count.pickle', 'wb') as handle:
        pickle.dump(frame_count, handle, protocol=pickle.HIGHEST_PROTOCOL)