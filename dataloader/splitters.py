import os
import pickle
import numpy as np
from scipy import io
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

class UCF101_splitter:
    def __init__(self, path, split):
        self.path = path
        self.split = split

    def get_action_index(self):
        self.action_label = {}
        with open(self.path + 'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label, action = line.split(' ')
            # print label,action
            if action not in self.action_label.keys():
                self.action_label[action] = label

    def split_video(self):
        self.get_action_index()
        for _, _, files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist' + self.split:
                    train_video = self.file2_dic(self.path + filename)
                if filename.split('.')[0] == 'testlist' + self.split:
                    test_video = self.file2_dic(self.path + filename)
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)

        return self.train_video, self.test_video

    def file2_dic(self, fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic = {}
        for line in content:
            # print line
            video = line.split('/', 1)[1].split(' ', 1)[0]
            key = video.split('_', 1)[1].split('.', 1)[0]
            label = self.action_label[line.split('/')[0]]
            dic[key] = int(label)
            # print key,label
        return dic

    def name_HandstandPushups(self, dic):
        dic2 = {}
        for video in dic:
            n, g = video.split('_', 1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_' + g
            else:
                videoname = video
            dic2[videoname] = dic[video]
        return dic2


class MyVideos_splitter:
    def __init__(self, frames_path):
        self.frames_path = frames_path

    def split_video(self):
        le = LabelEncoder()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        videos = np.array(sorted(os.listdir(self.frames_path)))
        labels = le.fit_transform([video.split('_')[0] for video in videos])
        for train_idx, valid_idx in sss.split(videos, labels):
            return videos[train_idx], videos[valid_idx]


class PennAction_splitter:
    def __init__(self, data_path, labels_path):
        self.data_path = data_path
        self.labels_path = labels_path

    def split_video(self):
        videos = np.array(sorted(os.listdir(self.data_path)))
        list_mat_files = sorted(os.listdir(self.labels_path))
        train = np.empty(len(list_mat_files), dtype=int)
        for i, mat_file in enumerate(list_mat_files):
            mat = io.loadmat(os.path.join(self.labels_path, mat_file))
            train[i] = int(mat['train'][0][0])
        train_idx = np.argwhere(train == 1).squeeze(axis=1)
        valid_idx = np.argwhere(train == -1).squeeze(axis=1)

        return videos[train_idx], videos[valid_idx]
