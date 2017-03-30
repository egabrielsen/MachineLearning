
import os
import sys
import random
import copy
import warnings
from random import randint
import _pickle as cPickle

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_X_y, check_array
from scipy.misc import imread
from scipy.special import expit
from skimage.filters import roberts
from skimage.feature import daisy
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score

def get_images(): # stuck this in a function to clean up memory
    dics = []
    for root, directory, files in os.walk('imgs'):
        for f in files:
            if 'data_batch' in f or 'test_batch' in f:
                with open(root+'/'+f, 'rb') as fo:
                    dics.append(cPickle.load(fo, encoding='latin1'))

    img_color = []
    img_labels = []
    for dic in dics:
        for i in range(len(dic['data'])):
            img_color.append(dic['data'][i]) # 1D img (1024 R, 1024 G, 1024 B)
            img_labels.append(dic['labels'][i]) # int representing the label

    img_color = np.array(img_color)
    img_labels = np.array(img_labels)

    # grab the mapping between label names and IDs
    print('Labels:')
    labels = {}
    with open('./imgs/batches.meta', 'rb') as fo:
        labels_tmp = cPickle.load(fo, encoding='latin1')
        for i in range(len(labels_tmp['label_names'])):
            labels[i] = labels_tmp['label_names'][i]
            print(i, "-->", labels_tmp['label_names'][i])
    print()

    img_label_names = np.array([labels[x] for x in img_labels])

    def toGrayscale(img):
        r, g, b = img[:1024], img[1024:2048], img[2048:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    img_gray = np.array([toGrayscale(x) for x in img_color])
    return (img_color, img_gray, img_labels, img_label_names)

img_color, img_gray, img_labels, img_label_names = get_images()
img_gray = img_gray
img_labels = img_labels
print("n_samples: {}".format(len(img_gray)))
print("n_features: {}".format(len(img_gray[0])))
print("n_classes: {}".format(len(np.unique(img_labels))))
print("Original Image Size: {} x {}".format(32, 32))




def global_contrast_normalization(x):
    x = x - x.mean(axis=1)[:, np.newaxis]
    normalizers = np.sqrt((x ** 2).sum(axis=1))
    x /= normalizers[:, np.newaxis]
    return x

normalized = np.array([np.concatenate(global_contrast_normalization(x.reshape((32, 32)))) for x in img_gray])
daisies = np.array([np.concatenate(np.concatenate(daisy(x.reshape((32,32)), step=16, radius=7, rings=2, histograms=8, orientations=5))) for x in img_gray])



class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=20, samp_percent=.25, replacement=False):
        super().__init__()
        self.n_estimators = n_estimators
        self.samp_percent = samp_percent
        self.replacement = replacement
        
    def _get_subset(self, x, y, replacement=True, samp_percent=.25, return_other=False):
        xy = np.hstack([x, y.reshape((len(y),1))])
        np.random.shuffle(xy)
        size = int(len(y)*samp_percent) if not replacement else len(y)
        indexes = np.random.choice(range(len(y)), size=size, replace=replacement)
        
        new_x = xy[indexes][:, :-1]
        new_y = xy[indexes][:, -1]
        
        if not return_other:
            return new_x, new_y
        else:
            inv_indexes = [i for i in range(len(xy)) if i not in indexes]
            alternate_x = xy[inv_indexes][:, :-1]
            alternate_y = xy[inv_indexes][:, -1]
            return new_x, new_y, alternate_x, alternate_y
        
        
    def fit(self, x, y, print_progress=False):
        x, y = check_X_y(x, y)
        x, y = x.copy(), y.copy()
        
        self.classifiers_ = []
        self.precisions_ = []
        self.n_features_ = x.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.classes_, y = np.unique(y, return_inverse=True)
        
        for i in range(self.n_estimators):
            if print_progress:
                sys.stdout.write('Estimator '+str(i)+' fitting.....')
            new_x, new_y, alt_x, alt_y = self._get_subset(x, y, return_other=True, replacement=self.replacement, samp_percent=self.samp_percent)
            c = MLPClassifier()
            c.fit(new_x, new_y)
            self.classifiers_.append(c)
            if print_progress:
                sys.stdout.write('testing.....')
            self.precisions_.append(f1_score(alt_y, c.predict(alt_x), average='weighted'))
            if print_progress:
                sys.stdout.write('complete!\n')
        
        return self
    
    def predict(self, x):
        x = check_array(x)
            
        results = np.zeros((x.shape[0], self.n_classes_))
        for classifier, precision in zip(self.classifiers_, self.precisions_):
            res = classifier.predict(x).astype(np.int)
            for i in range(len(res)):
                results[i][res[i]] += precision
        
        # select highest picked class for each row
        res = [np.argmax(x) for x in results]
        res = self.classes_[res]

        return res


p2 = Pipeline([('scaler', StandardScaler()), ('tlp', EnsembleClassifier(n_estimators=100, samp_percent=.15))])
cv = StratifiedShuffleSplit(n_splits=1, test_size=.2)
for train_index, test_index in cv.split(daisies, img_labels):
    p2.fit(daisies[train_index], img_labels[train_index], tlp__print_progress=True)
    yhat = p2.predict(daisies[test_index])
    print('f1 score:', f1_score(img_labels[test_index], yhat, average='macro'))
    print(np.bincount(yhat))
