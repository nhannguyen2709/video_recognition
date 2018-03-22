import os
import numpy as np

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, GlobalMaxPooling2D, GRU
from keras.optimizers import Adam

input_dir = 'data/NewVideos/jpegs_256/'

base_model = VGG19(weights='imagenet', include_top=False)
features = GlobalMaxPooling2D()(base_model.layers[-2].output)

vgg19_feature_extractor = Model(inputs=base_model.inputs, outputs=features)
vgg19_feature_extractor.summary()

num_videos = 53
tunable_num_frames = 250

for path, subdirs, files in os.walk(input_dir):

    for subdir in sorted(subdirs):
        path_to_video = os.path.join(input_dir, subdir)
        list_of_frames = sorted(os.listdir(path_to_video))
        num_frames = len(list_of_frames)
        input_to_vgg19 = np.zeros((tunable_num_frames, 224, 224, 3))
        
        for i, frame in enumerate(list_of_frames):
            try:
                img_path = os.path.join(path_to_video, frame)
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                input_to_vgg19[i, :, :, :] = x
            except IndexError:
                pass
        video_features = vgg19_feature_extractor.predict(input_to_vgg19, batch_size=50, verbose=1)
        print(video_features.shape)
        fname = os.path.join(subdir.replace('.mp4', '.out'))
        np.savetxt(os.path.join(path_to_video, fname), video_features)

y_train = []
x_train = np.zeros((53, 250, 512))
for path, subdirs, files in os.walk(input_dir):
    for i, subdir in enumerate(sorted(subdirs)):
        if 'PickUpObject' in subdir:
            y_train.append(0)
        elif 'TakeOffJacket' in subdir:
            y_train.append(1)
        else:
            y_train.append(2)
        path_to_video = os.path.join(input_dir, subdir)
        fname = os.path.join(subdir.replace('.mp4', '.out'))
        video_features = np.loadtxt(os.path.join(path_to_video, fname))
        x_train[i, :, :] = video_features

y_train = to_categorical(np.array(y_train))

video_inputs = Input((None, 512))
outputs = GRU(256, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)(video_inputs)
outputs = GRU(128, recurrent_dropout=0.2)(outputs)
outputs = Dense(3, activation='softmax')(outputs)
gru = Model(inputs=video_inputs, outputs=outputs)

gru.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'])

gru.fit(x_train, y_train, batch_size=9, epochs=5)
