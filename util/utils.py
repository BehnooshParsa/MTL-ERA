import matplotlib.pylab as plt
import matplotlib as mlt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import scipy
import scipy.io as sio
# import cv2
# from numba import jit, int64
import cv2
from collections import OrderedDict

plt.rcParams['figure.figsize'] = [10., 10.]


# ------------- Visualization -------------
def imshow_(x, **kwargs):
    ncls = 17
    classes = np.arange(1, ncls + 1)
    # print(classes)
    values = np.unique(classes.ravel())
    # print(values)
    col_list = ["light yellow", "gunmetal", "hot pink", "baby blue", "green blue", "baby pink", "blood red", "magenta",
                "dusky blue", "neon purple", "pale grey", "cool blue", "neon blue", "brown", "acid green", "orange",
                "viridian", "reddish orange", "dark magenta", "twilight blue"]
    col_list_palette = sns.xkcd_palette(col_list[0:17])
    CustomCmap = mlt.colors.ListedColormap(col_list_palette)
    # Set the palette using the name of a palette:
    # qualitative_colors = sns.color_palette("hls", ncls)
    # CustomCmap = mlt.colors.ListedColormap(qualitative_colors)
    if x.ndim == 2:
        im = plt.imshow(x, interpolation="nearest", cmap=CustomCmap, aspect=50)
    elif x.ndim == 1:
        im = plt.imshow(x[:, None].T, interpolation="nearest", cmap=CustomCmap, aspect=50)
        plt.yticks([])
    plt.axis("tight")
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="Class {l}".format(l=values[i])) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ------------- Data -------------
def mask_data(X, Y, Z, max_len=None, mask_value=0):
    if max_len is None:
        max_len = np.max([x.shape[0] for x in X])
    X_ = np.zeros([len(X), max_len, X[0].shape[1], X[0].shape[2]]) + mask_value
    Y_ = np.zeros([len(X), max_len]) + mask_value
    Z_ = np.zeros([len(X), max_len]) + mask_value

    # print(np.shape(Y[0]),np.shape(Y_))
    mask = np.zeros([len(X), max_len])
    for i in range(len(X)):
        l = X[i].shape[0]
        X_[i, :l,:,:] = X[i]
        Y_[i, :l] = Y[i]
        Z_[i, :l] = Z[i]
        mask[i, :l] = 1
    return X_, Y_,Z_, mask[:, :, None]
# def mask_data(X, Y, max_len=None, mask_value=0):
#     if max_len is None:
#         max_len = np.max([x.shape[0] for x in X])
#     X_ = np.zeros([len(X), max_len, X[0].shape[1]]) + mask_value
#     Y_ = np.zeros([len(X), max_len]) + mask_value
#
#     # print(np.shape(Y[0]),np.shape(Y_))
#     mask = np.zeros([len(X), max_len])
#     for i in range(len(X)):
#         l = X[i].shape[0]
#         X_[i, :l] = X[i]
#         Y_[i, :l] = Y[i]
#         mask[i, :l] = 1
#     return X_, Y_, mask[:, :, None]


def mask_data_single(X, max_len=None, mask_value=0):
    if max_len is None:
        max_len = np.max([x.shape[0] for x in X])
    X_ = np.zeros([len(X), max_len, X[0].shape[1]]) + mask_value

    mask = np.zeros([len(X), max_len])
    for i in range(len(X)):
        l = X[i].shape[0]
        X_[i, :l] = X[i]
        mask[i, :l] = 1
    return X_, mask[:, :, None]


# Unmask data
def unmask(X, M):
    if X[0].ndim == 1 or (X[0].shape[0] > X[0].shape[1]):
        return [X[i][M[i].flatten() > 0] for i in range(len(X))]
    else:
        return [X[i][:, M[i].flatten() > 0] for i in range(len(X))]


def match_lengths(X, Y, n_feat):
    # Check lengths of data and labels match
    if X[0].ndim == 1 or (X[0].shape[0] == n_feat):
        for i in range(len(Y)):
            length = min(X[i].shape[1], Y[i].shape[0])
            X[i] = X[i][:, :length]
            Y[i] = Y[i][:length]
    else:
        for i in range(len(Y)):
            length = min(X[i].shape[0], Y[i].shape[0])
            X[i] = X[i][:length]
            Y[i] = Y[i][:length]

    return X, Y


def remap_labels(Y_all):
    # Map arbitrary set of labels (e.g. {1,3,5}) to contiguous sequence (e.g. {0,1,2})
    ys = np.unique([np.hstack([np.unique(Y_all[i]) for i in range(len(Y_all))])])
    y_max = ys.max()
    y_map = np.zeros(y_max + 1, np.int) - 1
    for i, yi in enumerate(ys):
        y_map[yi] = i
    Y_all = [y_map[Y_all[i]] for i in range(len(Y_all))]
    return Y_all


def max_seg_count(Y):
    def seg_count(y):
        # Input label sequence
        return len(segment_labels(y))

    # Input: list of label sequences
    return max(map(seg_count, Y))


def subsample(X, Y, rate=1, dim=0):
    if dim == 0:
        X_ = [x[::rate] for x in X]
        Y_ = [y[::rate] for y in Y]
    elif dim == 1:
        X_ = [x[:, ::rate] for x in X]
        Y_ = [y[::rate] for y in Y]
    else:
        print("Subsample not defined for dim={}".format(dim))
        return None, None

    return X_, Y_


# def subsample(X, Y, rate=1, dim=1):
# 	if dim == 1:
# 		Y_ = [y[:,::rate] for y in Y]
# 		X_ = [cv2.resize(X[i].T, (X[i].shape[0], Y_[i].shape[1])).T for i in range(len(X))]
# 		# Y_ = [cv2.resize(y.T, (x.shape[0], T_new)).T for y in Y]
# 	elif dim == 0:
# 		Y_ = [y[::rate] for y in Y]
# 		X_ = [cv2.resize(X[i], (Y_[i].shape[0], X[i].shape[1])) for i in range(len(X))]
# 	else:
# 		print("Subsample not defined for dim={}".format(dim))
# 		return None, None

# 	return X_, Y_

# ------------- Segment functions -------------
def segment_labels(Yi):
    Yi = Yi.squeeze()
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]).tolist() + [len(Yi)] # we had a +1 in the nonzero I think because of the -1 label.
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split


def segment_data(Xi, Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Xi_split = [np.squeeze(Xi[:, idxs[i]:idxs[i + 1]]) for i in range(len(idxs) - 1)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Xi_split, Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    return intervals


def segment_lengths(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i + 1] - idxs[i]) for i in range(len(idxs) - 1)]
    return np.array(intervals)




# ------------- IO -------------
def save_predictions(dir_out, y_pred, y_truth, idx_task, experiment_name=""):
    if experiment_name != "":
        dir_out += "/{}/".format(experiment_name)
    # Make sure fiolder exists
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    truth_test_all_out = {"t{}_{}".format(idx_task, k): v for (k, v) in enumerate(y_truth)}
    predict_test_all_out = {"t{}_{}".format(idx_task, k): v for k, v in enumerate(y_pred)}
    sio.savemat(dir_out + "/{}_truth.mat".format(idx_task), truth_test_all_out)
    sio.savemat(dir_out + "/{}_predict.mat".format(idx_task), predict_test_all_out)


# ------------- Vision -------------
def load_images(uris, rez_im, uri_data):
    # Load images for CNN
    X = np.empty((len(uris), 3, rez_im, rez_im), dtype=np.float32)
    for i, x in enumerate(uris):
        im = cv2.imread(uri_data + x)
        im = cv2.resize(im, (rez_im, rez_im))
        X[i] = im.T
    return X


def check_images_available(x_uri, y, uri_data):
    # Check if there are any missing files
    no_file = []
    for i, x in enumerate(x_uri):
        if not os.path.isfile(uri_data + x):
            # print("Missing", x)
            no_file += [i]
    x_uri = np.array([x_uri[i] for i in range(len(x_uri)) if i not in no_file])
    y = np.array([y[i] for i in range(len(y)) if i not in no_file])

    if len(no_file) > 0:
        print("Missing #", len(no_file))

    return x_uri, y

# ----------------- Metrics ----------------
class ComputeMetrics:
    metric_types = ["accuracy", "edit_score", "overlap_f1"]
    # metric_types = ["macro_accuracy", "acc_per_class"]
    # metric_types += ["classification_accuracy"]
    # metric_types += ["precision", "recall"]
    # metric_types += ["mAP1", "mAP5", "midpoint"]
    trials = []

    def __init__(self, metric_types=None, overlap=.1, bg_class=None, n_classes=None):
        if metric_types is not None:
            self.metric_types = metric_types

        self.scores = OrderedDict()
        self.attrs = {"overlap": overlap, "bg_class": bg_class, "n_classes": n_classes}
        self.trials = []

        for m in self.metric_types:
            self.scores[m] = OrderedDict()

    @property
    def n_classes(self):
        return self.attrs['n_classes']

    def set_classes(self, n_classes):
        self.attrs['n_classes'] = n_classes

    def add_predictions(self, trial, P, Y):
        if trial not in self.trials:
            self.trials += [trial]

        for m in self.metric_types:
            self.scores[m][trial] = globals()[m](P, Y, **self.attrs)

    def print_trials(self, metric_types=None):
        if metric_types is None:
            metric_types = self.metric_types

        for trial in self.trials:
            scores = [self.scores[m][trial] for m in metric_types]
            scores_txt = []
            for m, s in zip(metric_types, scores):
                if type(s) is np.float64:
                    scores_txt += ["{}:{:.04}".format(m, s)]
                else:
                    scores_txt += [("{}:[".format(m) + "{:.04}," * len(s)).format(*s) + "]"]
            # txt = "Trial {}: ".format(trial) + " ".join(["{}:{:.04}".format(metric_types[i], scores[i]) for i in range(len(metric_types))])
            txt = "Trial {}: ".format(trial) + ", ".join(scores_txt)
            print(txt)

    def print_scores(self, metric_types=None):
        if metric_types is None:
            metric_types = self.metric_types

        scores = [np.mean([self.scores[m][trial] for trial in self.trials]) for m in metric_types]
        txt = "All: " + " ".join(["{}:{:.04}".format(metric_types[i], scores[i]) for i in range(len(metric_types))])
        print(txt)


def accuracy(P, Y, **kwargs):
    def acc_(p, y):
        return np.mean(p == y) * 100

    if type(P) == list:
        return np.mean([np.mean(P[i] == Y[i]) for i in range(len(P))]) * 100
    else:
        return acc_(P, Y)


def levenstein_(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(P, Y, norm=True, bg_class=None, **kwargs):
    if type(P) == list:
        tmp = [edit_score(P[i], Y[i], norm, bg_class) for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_ = segment_labels(P)
        Y_ = segment_labels(Y)
        if bg_class is not None:
            P_ = [c for c in P_ if c != bg_class]
            Y_ = [c for c in Y_ if c != bg_class]
        return levenstein_(P_, Y_, norm)


def overlap_f1(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p, y, n_classes, bg_class, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels != bg_class]
            true_labels = true_labels[true_labels != bg_class]
            pred_intervals = pred_intervals[pred_labels != bg_class]
            pred_labels = pred_labels[pred_labels != bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) - np.maximum(pred_intervals[j, 0],
                                                                                               true_intervals[:, 0])
            union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(pred_intervals[j, 0],
                                                                                        true_intervals[:, 0])
            IoU = (intersection / union) * (pred_labels[j] == true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1

        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1 * 100

    if type(P) == list:
        return np.mean([overlap_(P[i], Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)
