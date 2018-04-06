import os
import pickle

ucf101 = pickle.load(
    open('dataloader/dic/ucf101_frame_count.pickle', mode='rb'))
merged = pickle.load(
    open('dataloader/dic/merged_frame_count.pickle', mode='rb'))


valid_video_names = sorted(os.listdir('data/NewVideos/validation_videos/'))
valid_video_paths = [os.path.join(
    'data/NewVideos/validation_videos/', video_name) for video_name in valid_video_names]
valid_video_frame_counts = [len(sorted(os.listdir(video_path)))
                            for video_path in valid_video_paths]

valid_video_frame_count = {}
for video_name, video_frame_count in zip(valid_video_names, valid_video_frame_counts):
    old_count = ucf101[video_name+'.avi']
    valid_video_frame_count[video_name] = video_frame_count
    if old_count != video_frame_count:
        print('\nUCF101 count of {} is {}'.format(video_name, old_count))
        print('Count from directory is {}'.format(video_frame_count))
        ucf101[video_name+'.avi'] = video_frame_count
    else:
        merged[video_name] = video_frame_count

with open('dataloader/dic/ucf101_frame_count.pickle', 'wb') as handle:
    pickle.dump(ucf101, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('dataloader/dic/merged_frame_count.pickle', 'wb') as handle:
    pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
