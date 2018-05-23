import os, shutil
from tqdm import tqdm
from splitters import UCF101_splitter, PennAction_splitter, MyVideos_splitter


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(
                    s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


def split_ucf101_dataset(split='01'):
    frames_path = '../data/UCF101/frames'
    u_path = '../data/UCF101/tvl1_flow/u'
    v_path = '../data/UCF101/tvl1_flow/v'

    splitter = UCF101_splitter(path='../UCF_list/', split=split)
    train_videos, test_videos = splitter.split_video()
    print('Train videos: {}, validation videos: {}'.format(
          len(train_videos),
          len(test_videos)))

    train_frames_dst = '../data/UCF101/train/frames'
    test_frames_dst = '../data/UCF101/test/frames'
    train_u_dst = '../data/UCF101/train/tvl1_flow/u'
    test_u_dst = '../data/UCF101/test/tvl1_flow/u'
    train_v_dst = '../data/UCF101/train/tvl1_flow/v'
    test_v_dst = '../data/UCF101/test/tvl1_flow/v'

    for train_video in tqdm(sorted(train_videos.keys())):
        train_video_name = 'v_' + train_video
        copytree(os.path.join(frames_path, train_video_name),
                 os.path.join(train_frames_dst, train_video_name))
        copytree(os.path.join(u_path, train_video_name),
                 os.path.join(train_u_dst, train_video_name))
        copytree(os.path.join(v_path, train_video_name),
                 os.path.join(train_v_dst, train_video_name))

    for test_video in tqdm(sorted(test_videos.keys())):
        test_video_name = 'v_' + test_video
        copytree(os.path.join(frames_path, test_video_name),
                 os.path.join(test_frames_dst, test_video_name))
        copytree(os.path.join(u_path, test_video_name),
                 os.path.join(test_u_dst, test_video_name))
        copytree(os.path.join(v_path, test_video_name),
                 os.path.join(test_v_dst, test_video_name))


def split_my_videos_dataset():
    frames_path = '../data/MyVideos/frames'
    poses_path = '../data/MyVideos/poses'
    splitter = MyVideos_splitter(frames_path=frames_path)
    train_videos, test_videos = splitter.split_video()
    print(
        'Train videos: {}, validation videos: {}'.format(
            len(train_videos),
            len(test_videos)))
    
    train_frames_dst =  '../data/MyVideos/train/frames'
    test_frames_dst = '../data/MyVideos/validation/frames'
    train_poses_dst = '../data/MyVideos/train/poses'
    test_poses_dst = '../data/MyVideos/validation/poses'

    for train_video in tqdm(train_videos):
        copytree(os.path.join(frames_path, train_video),
                 os.path.join(train_frames_dst, train_video))
        copytree(os.path.join(poses_path, train_video),
                 os.path.join(train_poses_dst, train_video))
    
    for test_video in tqdm(test_videos):
        copytree(os.path.join(frames_path, test_video),
                 os.path.join(test_frames_dst, test_video))
        copytree(os.path.join(poses_path, test_video),
                 os.path.join(test_poses_dst, test_video))


def split_penn_action_dataset():
    data_path='../data/Penn_Action/frames'
    labels_path='../data/Penn_Action/labels'
    splitter = PennAction_splitter(data_path=data_path,
                                   labels_path=labels_path)
    train_videos, test_videos = splitter.split_video()
    print(
        'Train videos: {}, validation videos: {}'.format(
            len(train_videos),
            len(test_videos)))

    train_frames_dst = '../data/Penn_Action/train/frames'
    test_frames_dst = '../data/Penn_Action/validation/frames'
    train_labels_dst = '../data/Penn_Action/train/labels'
    test_labels_dst = '../data/Penn_Action/validation/labels'

    for train_video in tqdm(train_videos):
        train_video_mat = train_video + '.mat'
        copytree(os.path.join(data_path, train_video),
                 os.path.join(train_frames_dst, train_video))

        shutil.copy2(os.path.join(labels_path, train_video_mat),
                    os.path.join(train_labels_dst, train_video_mat))

    for test_video in tqdm(test_videos):
        test_video_mat = test_video + '.mat'
        copytree(os.path.join(data_path, test_video),
                 os.path.join(test_frames_dst, test_video))

        shutil.copy2(os.path.join(labels_path, test_video_mat),
                    os.path.join(test_labels_dst, test_video_mat))


if __name__ == '__main__':
    split_ucf101_dataset()