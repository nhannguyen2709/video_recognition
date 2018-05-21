import itertools
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np

from dataloader.keras_data import PennAction, MyVideos
from keras.models import load_model

from sklearn.metrics import accuracy_score, confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Model and data
model = load_model('checkpoint/penn_action.hdf5')
valid_videos = PennAction(
        frames_path='data/PennAction/validation/frames',
        labels_path='data/PennAction/validation/labels',
        batch_size=8,
        num_frames_sampled=16,
        shuffle=False)

# Predict
num_sampling_times = 25
num_classes = valid_videos.num_classes
y_true = valid_videos.y
class_names = valid_videos.labels

y_pred = np.zeros((len(valid_videos.y), num_classes))
for i in range(num_sampling_times):
    y_pred += model.predict_generator(valid_videos, workers=4, verbose=1)
y_pred /= num_sampling_times
y_pred = np.argmax(y_pred, axis=1)
np.save('checkpoint/penn_action_pred.npy', y_pred)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('checkpoint/penn_action_cnf_matrix.png')
plt.show()
