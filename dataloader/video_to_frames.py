import time
import cv2
import os
import pickle


def video_to_frames(input_path, output_path):
    """Function to extract frames from an input video file
    and save them as separate frames in an output directory.
    Args:
        input_path: Path to video file.
        output_path: Path to save the frames.
    Returns:
        Video filename and its frame count
    """
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_path)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("\nNumber of frames: ", video_length)
    count = 0
    print("Converting video...")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        _, frame = cap.read()
        # Write the results back to output location
        cv2.imwrite(output_path + "/frame%#06d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames from {}".format(
                input_path.split('/')[-1]))
            print("Took {} seconds for converting".format(
                round(time_end-time_start, 4)))
    return video_length


def process_multiple_videos(path_to_videos, path_to_videos_frames, path_to_frame_count):
    # Create a frame count for videos
    frame_count = {}

    if not os.path.exists(path_to_videos_frames):
        os.mkdir(path_to_videos_frames)

    for path, _, filenames in os.walk(path_to_videos):
        for filename in sorted(filenames):
            path_to_video = os.path.join(path, filename)
            if path_to_video.endswith('.mp4'):
                video_filename = os.path.splitext(filename)[0]
                output_loc = os.path.join(
                    path_to_videos_frames, video_filename)
                if not os.path.exists(output_loc):
                    os.mkdir(output_loc)
                # Extract frames, return video filename and its frame count
                video_length = video_to_frames(path_to_video, output_loc)
                frame_count[video_filename] = video_length
            else:
                pass

    # Dump the frame count dict into pickle file
    with open(path_to_frame_count, 'wb') as handle:
        pickle.dump(frame_count, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_single_video(path_to_video, path_to_video_frames, path_to_frame_count):
    filename = path_to_video.split('/')[-1]
    video_filename = os.path.splitext(filename)[0]
    frame_count = pickle.load(open(path_to_frame_count, 'rb'))
    output_loc = os.path.join(path_to_video_frames, video_filename)

    if not os.path.exists(output_loc):
        os.mkdir(output_loc)

    video_length = video_to_frames(path_to_video, output_loc)
    frame_count[video_filename] = video_length

    with open(path_to_frame_count, 'wb') as handle:
        pickle.dump(frame_count, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    process_single_video('../data/NewVideos/train_videos_multiple_actions/v_MultipleActions_g01_c01.MP4',
                         '../data/NewVideos/videos_frames_multiple_actions',
                         '../dataloader/dic/new_videos_frame_count.pickle')

    # process_multiple_videos('../data/NewVideos/train_videos',
    #                         '../data/NewVideos/videos_frames',
    #                         '../dataloader/dic/new_videos_frame_count.pickle')
