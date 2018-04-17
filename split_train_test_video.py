import shutil
from tqdm import tqdm
from dataloader.splitters import UCF101_splitter, PennAction_splitter

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def split_ucf101_dataset():
    splits = ['01', '02', '03']
    path = 'UCF_list/'

    for split in splits:
        splitter = UCF101_splitter(path=path,split=split)
        train_video,test_video = splitter.split_video()
        print(len(train_video),len(test_video))

        train_dst = 'data/train_videos_' + split
        test_dst = 'data/test_videos_' + split

        for train_video in tqdm(sorted(train_video.keys())):
            train_video_name = 'v_' + train_video
            copytree(os.path.join('data/UCF101/jpegs_256/', train_video_name), 
                     os.path.join(train_dst, train_video_name))

        for test_video in tqdm(sorted(test_video.keys())):
            test_video_name = 'v_' + test_video
            copytree(os.path.join('data/UCF101/jpegs_256/', test_video_name), 
                     os.path.join(test_dst, test_video_name))
            
def split_penn_action_dataset():
    splitter = PennAction_splitter(data_path='data/Penn_Action/frames',
                                   labels_path='data/Penn_Action/labels')
    train_videos,test_videos = splitter.split_video()
    print('Train videos: {}, validation videos: {}'.format(len(train_video),len(test_video)))

    train_frames_dst = 'data/Penn_Action/train/frames'
    test_frames_dst = 'data/Penn_Action/validation/frames'
    train_labels_dst = 'data/Penn_Action/train/labels'
    test_labels_dst = 'data/Penn_Action/validation/labels'

    for train_video in tqdm(train_videos):
        train_video_mat = train_video + '.mat'
        copytree(os.path.join(data_path, train_video), 
                 os.path.join(train_frames_dst, train_video))   
        
        copytree(os.path.join(labels_path, train_video_mat), 
                 os.path.join(train_labels_dst, train_video_mat))       

    for test_video in tqdm(test_videos):
        test_video_mat = test_video + '.mat'
        copytree(os.path.join(data_path, test_video), 
                 os.path.join(test_frames_dst, test_video))
        
        copytree(os.path.join(labels_path, test_video_mat), 
                 os.path.join(test_labels_dst, test_video_mat))
        
    
if __name__=='__main__':
    split_penn_action_dataset()