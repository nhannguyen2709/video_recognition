import os, pickle


class UCF101_splitter():
    def __init__(self, path, split):
        self.path = path
        self.split = split

    def get_action_index(self):
        self.action_label={}
        with open(self.path+'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label,action = line.split(' ')
            #print label,action
            if action not in self.action_label.keys():
                self.action_label[action]=label

    def split_video(self):
        self.get_action_index()
        for path, subdir,files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    train_video = self.file2_dic(self.path+filename)
                if filename.split('.')[0] == 'testlist'+self.split:
                    test_video = self.file2_dic(self.path+filename)
        print('==> Training videos: {}, Validation videos: {}'.format(len(train_video), len(test_video)))
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)

        return self.train_video, self.test_video

    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic={}
        for line in content:
            #print line
            video = line.split('/',1)[1].split(' ',1)[0]
            key = video.split('_',1)[1].split('.',1)[0]
            label = self.action_label[line.split('/')[0]]   
            dic[key] = int(label)
            #print key,label
        return dic

    def name_HandstandPushups(self,dic):
        dic2 = {}
        for video in dic:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            else:
                videoname=video
            dic2[videoname] = dic[video]
        return dic2

if __name__ == '__main__':
    import shutil
    from tqdm import tqdm

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

    splits = ['01', '02', '03']
    path = '../UCF_list/'
    
    for split in splits:
        splitter = UCF101_splitter(path=path,split=split)
        train_video,test_video = splitter.split_video()
        print(len(train_video),len(test_video))
        
        train_dst = '../data/train_videos_' + split
        test_dst = '../data/test_videos_' + split

        for train_video in tqdm(sorted(train_video.keys())):
            train_video_name = 'v_' + train_video
            copytree(os.path.join('../data/UCF101/jpegs_256/', train_video_name), 
                     os.path.join(train_dst, train_video_name))

        for test_video in tqdm(sorted(test_video.keys())):
            test_video_name = 'v_' + test_video
            copytree(os.path.join('../data/UCF101/jpegs_256/', test_video_name), 
                     os.path.join(test_dst, test_video_name))
