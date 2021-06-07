import numpy as np
from torch.utils import data
# from termcolor import colored
import yaml
try:
    with open('./config_files/config_UW_data.yml', 'r') as file:
        config_data = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')

def readtxt(fnm):
    with open(fnm, 'r') as d:
        dataf = d.readlines()
    return dataf


end_frame = [1472, 1517, 1145, 1810, 1446, 1703, 1432, 1865, 1535, 1830, 2372, 2316, 1595, 2130, 1601, 2479, 1866, 1593,
             1935, 1404]
start_frame = [49, 72, 41, 55, 66, 67, 105, 83, 77, 77, 82, 83, 115, 62, 54, 98, 76, 63, 94, 54]
joints = ['0: right ankle', '1: left ankle', '2: right knee', '3: left knee', '4: right hip',
          '5: left hip', '6: right wrist', '7: left wrist', '8: right elbow', '9: left elbow',
          '10: right shoulder', '11: left shoulder', '12: neck', '13: head', '14: center']

pairs = {'0: left upperarm': (9, 11), '1: left forearm': (7, 9),
         '2: left shin': (1, 3), '3: left thigh': (3, 5),  # bones on the left
         '4: right shin': (0, 2), '5: right thigh': (2, 4),
         '6: right upperarm': (8, 10), '7: right forearm': (6, 8),  # bones on the right
         '8: hip': (4, 5), '9: shoulder': (10, 11),
         '10: neck': (12, 13)}  # bones on the torso


class Dataset_with_REBA(data.Dataset):
    def __init__(self, seqlist, history=90):
        self.seqlist = seqlist
        self.poselist = []
        self.labellist = []
        self.rebascorelist = []
        self.history = history
        self.number_of_seq = len(seqlist)
        self.mask = 0
        base_dir  = config_data['base_data_dir']
        threed_poseloc = base_dir+config_data['threed_poseloc']
        labelloc = base_dir+config_data['label_dir']
        labelnames = list(np.load(base_dir+config_data['labelnames']))
        reba_scores_loc = base_dir+config_data['reba_scores_loc']
        
        print('Processing Data ...')
        for seq in seqlist:

            # joint_features = pkl.load(open(jointfeaturesloc + 'node_attr_' + seq + '_pose3d', "rb"))['node_glob_pos'][:,
            #                  :-1, :]
            # threepose = pkl.load(open(jointfeaturesloc + 'node_attr_' + seq + '_pose3d', "rb"))[
            #     'node_glob_pos']  #
            threepose = np.load(threed_poseloc + seq + '_pose3d.npy')
            labels = np.load(labelloc + seq + '.npy')
            rebascore = np.loadtxt(reba_scores_loc + seq + '.txt')
            # labels = label2index(labels_txt, labelnames)
            # if seq=='15':
            #     print(seq)
            if len(labels) != np.shape(threepose)[0]:
                raise Exception('Labels and data dimension does not match in video {}'.format(seq))

            # threepose = threepose[0:len(labels)]
            if self.history != None:
                threepose = self.truncatewithhistory(threepose)
                # print(colored('Size of input: '+ str(threepose[-1].shape), 'blue'))
                labels = self.truncatewithhistory(labels)
                # print(colored('Size of output labels: ' + str(len(labels[-1])), 'blue'))
                rebascore = self.truncatewithhistory(rebascore)
                # print(colored('Size of output reba scores: ' + str(rebascore[-1].shape), 'blue'))

                for xx in threepose:
                    self.poselist.append(xx)

                for xx in labels:
                    self.labellist.append(xx)

                for xx in rebascore:
                    self.rebascorelist.append(xx)
            else:
                self.poselist.append(threepose)

                self.labellist.append(np.array(labels))

                self.rebascorelist.append(rebascore)


        if self.history == None:
            lengths = [x.shape[0] for x in self.poselist]
            self.max_len = np.max(lengths)

    def __len__(self):
        return len(self.poselist)

    def truncatewithhistory(self, x):
        x_trunk = []
        # y_trunk = []
        for i in range(0, len(x), self.history):
            xt = x[i:i + self.history]
            # yt = y[i:i + self.history]
            if len(xt) == self.history:
                x_trunk.append(xt)
                # y_trunk.append(yt)
            else:
                xt = x[-self.history:]
                # yt = y[-self.history:]
                x_trunk.append(xt)
                # y_trunk.append(yt)

        return x_trunk

    def mask_data(self, max_len, mask_value=0):
        self.max_len = max_len
        X_ = np.zeros(
            [self.number_of_seq, self.max_len, self.poselist[0].shape[1], self.poselist[0].shape[2]]) + mask_value
        Y_ = np.zeros([self.number_of_seq, self.max_len]) + mask_value
        Z_ = np.zeros([self.number_of_seq, self.max_len]) + mask_value

        # print(np.shape(Y[0]),np.shape(Y_))
        mask = np.zeros([self.number_of_seq, self.max_len])
        for i in range(self.number_of_seq):
            l = self.poselist[i].shape[0]
            X_[i, :l, :, :] = self.poselist[i]
            Y_[i, :l] = self.labellist[i]
            Z_[i, :l] = self.rebascorelist[i]
            mask[i, :l] = 1
        self.poselist = X_
        self.labellist = Y_
        self.rebascorelist = Z_
        self.mask = mask[:, :, None]

    def __getitem__(self, index):
        pose = self.poselist[index]
        lab = self.labellist[index]
        reba = self.rebascorelist[index]
        # print(pose.shape, np.asarray(lab).shape)
        return np.double(pose), np.double(np.asarray(lab)), np.double(reba)


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def adjacencyMat(pairs, nodeNum):
    adj_mat = np.eye(nodeNum)
    for p in pairs:
        i = pairs[p][0]
        j = pairs[p][1]
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1
    return adj_mat


#
def hop_create():
    hop = [
        [0, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, 0, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [1, np.inf, 0, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, 1, np.inf, 0, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, 1, np.inf, 0, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1],
        [np.inf, np.inf, np.inf, 1, np.inf, 0, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, 1],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 0, np.inf, 1, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 0, np.inf, 1, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, 0, np.inf, 1, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, 0, 1, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, 0, 1, 1],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 0, np.inf],
        [np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 0]]
    return hop


#
def createnewAs(AN, hop_dis, num_node=15):
    A_list = []
    for hop in range(0, 2):
        a_root = np.zeros((num_node, num_node))
        a_close = np.zeros((num_node, num_node))
        a_further = np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if hop_dis[j, i] == hop:
                    if hop_dis[j, 1] == hop_dis[i, 1]:
                        a_root[j, i] = AN[j, i]
                    elif hop_dis[j, 1] > hop_dis[i, 1]:
                        a_close[j, i] = AN[j, i]
                    else:
                        a_further[j, i] = AN[j, i]
        if hop == 0:
            A_list.append(a_root)
        else:
            A_list.append(a_root + a_close)
            A_list.append(a_further)
    As = np.stack(A_list)
    return As


#
#
def getAs():
    A = adjacencyMat(pairs, 15)
    AN = normalize_digraph(A)
    hop_dis = np.asarray(hop_create())
    As = createnewAs(AN, hop_dis, num_node=15)
    return As


def label2index(ms, labelnms):
    indexlist = []
    for m in ms:
        idx = labelnms.index(m)
        indexlist.append(idx)
    return indexlist
#
#
#
# def labelfinder():
#     labelloc = '/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/Labels/'
#     txtfiles = glob.glob(labelloc + '*.txt')
#     labelnmlist = []
#     for t in txtfiles:
#         with open(t, 'r') as f:
#             lab = f.readlines()
#             for l in lab:
#                 labelnmlist.append(l)
#     np.save('/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/uniquelabelnames.npy',
#             list(np.unique(labelnmlist)))
#     print('Done')
#
#
# class Dataset_offline(data.Dataset):
#     def __init__(self, X, Y):
#         self.poselist = X
#         self.labellist = Y
#
#     def __len__(self):
#         return len(self.poselist)
#
#     def __getitem__(self, index):
#         pose = self.poselist[index]
#         lab = self.labellist[index]
#         # print(pose.shape, np.asarray(lab).shape)
#         return np.double(pose), np.double(np.asarray(lab))
#
#
# class Dataset:
#     name = ""
#     n_classes = None
#     n_features = None
#     activity = None
#
#     def __init__(self, name="", base_dir="", activity=None):
#         self.name = name
#         self.base_dir = os.path.expanduser(base_dir)
#
#         # Find the number of splits
#         split_folders = os.listdir('/home/smartlab/Documents/Dataset/{}/EDTCN/Splits/'.format(self.name))
#         self.splits = np.sort([s for s in split_folders if "split" in s])
#         self.n_splits = len(self.splits)
#
#     def feature_path(self, features):
#         return os.path.expanduser('/home/smartlab/Documents/Dataset/UW_IOM/EDTCN/SuperConcatFeatures/')
#         # return os.path.expanduser('/home/smartlab/Documents/Dataset/UW_IOM/Poses/LCRnet/concatFeatures/')
#         # return os.path.expanduser(self.base_dir+"Features/{}/{}/".format(self.name, features))
#
#     def get_files(self, dir_features, split=None):
#         if "split_1" in os.listdir(dir_features):
#             files_features = np.sort(os.listdir(dir_features + "/{}/".format(split)))
#         else:
#             files_features = np.sort(os.listdir(dir_features))
#
#         # files_features = [f for f in files_features if f.find(".mat")>=0]
#         files_features = [f for f in files_features if f.find(".npy") >= 0]
#         #         print('files_features',files_features)
#         return files_features
#
#     def fid2idx(self, files_features, extensions=[".mov", ".mat", ".avi", "rgb-", ".npy"]):
#         return {remove_exts(files_features[i], extensions): i for i in range(len(files_features))}
#
#     def load_split(self, features, split, sample_rate=0.5):
#         # Setup directory and filenames
#         dir_features = self.feature_path(features)
#
#         # Get splits for this partion of data
#         if self.activity == None:
#             file_train = open(
#                 "/home/smartlab/Documents/Dataset/{}/EDTCN/Splits/{}/train.txt".format(self.name, split)).readlines()
#             file_test = open(
#                 "/home/smartlab/Documents/Dataset/{}/EDTCN/Splits/{}/test.txt".format(self.name, split)).readlines()
#         else:
#             file_train = open(
#                 "/home/smartlab/Documents/Dataset/UW_IOM/EDTCN/splits/{}/{}/{}/train.txt".format(self.name,
#                                                                                                  self.activity,
#                                                                                                  split)).readlines()
#             file_test = open(
#                 "/home/smartlab/Documents/Dataset/UW_IOM/EDTCN/{}/{}/{}/test.txt".format(self.name, self.activity,
#                                                                                          split)).readlines()
#
#             # / home / idwivedi / PycharmProjects / UW_IOM_Dataset / EDTCN / splits / UW_IOM / split_1
#         file_train = [f.strip() for f in file_train]
#         file_test = [f.strip() for f in file_test]
#         # Remove extension
#         if "." in file_train[0]:
#             file_train = [".".join(f.split(".")[:-1]) for f in file_train]
#             file_test = [".".join(f.split(".")[:-1]) for f in file_test]
#
#         self.trials_train = file_train
#         self.trials_test = file_test
#
#         # Get all features
#         files_features = self.get_files(dir_features, split)
#         X_all, Y_all = [], []
#         dir_labels = '/home/smartlab/Documents/Dataset/UW_IOM/EDTCN/ClassNum/'
#         for ii, f in enumerate(files_features):
#             if "split_" in os.listdir(dir_features)[-1]:
#                 # data_tmp = sio.loadmat( closest_file("{}{}/{}".format(dir_features,split, f)) )
#                 # print(closest_file("{}{}/{}".format(dir_labels,split, f)))
#                 data_tmp_Y = np.loadtxt(closest_file("{}{}/{}".format(dir_labels, split, str(ii + 1) + '.txt')))[1:]
#                 data_tmp_X = np.load(closest_file("{}{}/{}".format(dir_features, split, f)))
#                 # print('here')
#                 # print('data_tmp_X: ',np.shape(data_tmp_X))
#                 # print('data_tmp_Y: ',np.shape(data_tmp_Y))
#
#
#             else:
#                 # data_tmp = sio.loadmat( closest_file("{}/{}".format(dir_features, f)) )
#                 print('here')
#                 # data_tmp_Y = np.load(closest_file("{}/{}".format(dir_labels, f)))
#                 # data_tmp_X = np.load( closest_file("{}/{}".format(dir_features, f)))
#
#             X_all += [data_tmp_X.astype(np.float32)]
#             Y_all += [data_tmp_Y.astype(int)]
#             # print('data_tmp_Y: ',np.shape(data_tmp_Y))
#             # print(f)
#             # print('X_all ', np.shape(X_all[-1]))
#             # print('Y_all ', np.shape(Y_all[-1]))
#
#         # Make sure axes are correct (TxF not FxT for F=feat, T=time)
#         print("Make sure axes are correct (TxF not FxT for F=feat, T=time)")
#         print(X_all[0].shape, Y_all[0].shape)
#         if X_all[0].shape[0] != Y_all[0].shape[0]:
#             X_all = [x.T for x in X_all]
#         self.n_features = X_all[0].shape[1]
#         # print('n_features: ',self.n_features)
#         self.n_classes = np.size(np.unique(np.concatenate(Y_all)))  # len(np.unique(np.vstack(Y_all)))
#         # print('X_all',np.shape(X_all[0]))
#         # Make sure labels are sequential
#         # print('n_classes: ',self.n_classes)
#         # print(np.shape(np.hstack(Y_all)))
#
#         if self.n_classes != int(np.concatenate(Y_all).max()) + 1:
#             Y_all = remap_labels(Y_all)
#             print("Reordered class labels")
#
#         # Subsample the data
#         if sample_rate > 1:
#             # print('sample_rate',sample_rate)
#             X_all, Y_all = subsample(X_all, Y_all, sample_rate, dim=0)
#             # print('X_all',np.shape(X_all[0]))
#         # print('Y_all',np.shape(Y_all[0]))
#         # ------------Train/test Splits---------------------------
#         # Split data/labels into train/test splits
#         fid2idx = self.fid2idx(files_features)
#         # print(file_train)
#         # print(fid2idx)
#         X_train = [X_all[int(f) - 1] for f in file_train if int(f) - 1 in fid2idx.values()]
#         X_test = [X_all[int(f) - 1] for f in file_test if int(f) - 1 in fid2idx.values()]
#         # print(len(X_train))
#         y_train = [Y_all[int(f) - 1] for f in file_train if int(f) - 1 in fid2idx.values()]
#         y_test = [Y_all[int(f) - 1] for f in file_test if int(f) - 1 in fid2idx.values()]
#         # print('Xtrain', np.shape(X_train))
#         # print('Ytrain', np.shape(y_train))
#         # print('Xtest', np.shape(X_test))
#         # print('Ytest', np.shape(y_train))
#         if len(X_train) == 0:
#             print("Error loading data")
#
#         return X_train, y_train, X_test, y_test
#

# def to_categorical(y, num_classes=None, dtype='float32'):
#     """Converts a class vector (integers) to binary class matrix.
#
#     E.g. for use with categorical_crossentropy.
#
#     # Arguments
#         y: class vector to be converted into a matrix
#             (integers from 0 to num_classes).
#         num_classes: total number of classes.
#         dtype: The data type expected by the input, as a string
#             (`float32`, `float64`, `int32`...)
#
#     # Returns
#         A binary matrix representation of the input. The classes axis
#         is placed last.
#     """
#     y = np.array(y, dtype='int')
#     input_shape = y.shape
#     if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
#         input_shape = tuple(input_shape[:-1])
#     y = y.ravel()
#     if not num_classes:
#         num_classes = np.max(y) + 1
#     n = y.shape[0]
#     categorical = np.zeros((n, num_classes), dtype=dtype)
#     categorical[np.arange(n), y] = 1
#     output_shape = input_shape + (num_classes,)
#     categorical = np.reshape(categorical, output_shape)
#     return categorical
#
#
# def remap_labels(Y_all):
#     # Map arbitrary set of labels (e.g. {1,3,5}) to contiguous sequence (e.g. {0,1,2})
#     ys = np.unique([np.hstack([np.unique(Y_all[i]) for i in range(len(Y_all))])])
#     y_max = ys.max()
#     y_map = np.zeros(y_max + 1, np.int) - 1
#     for i, yi in enumerate(ys):
#         y_map[yi] = i
#     Y_all = [y_map[Y_all[i]] for i in range(len(Y_all))]
#     return Y_all
#
#
# def subsample(X, Y, rate=1, dim=0):
#     if dim == 0:
#         X_ = [x[::rate] for x in X]
#         Y_ = [y[::rate] for y in Y]
#     elif dim == 1:
#         X_ = [x[:, ::rate] for x in X]
#         Y_ = [y[::rate] for y in Y]
#     else:
#         print("Subsample not defined for dim={}".format(dim))
#         return None, None
#
#     return X_, Y_
#
#
# def closest_file(fid, extension=".mat"):
#     # Fix occasional issues with extensions (e.g. X.mp4.mat)
#     basename = os.path.basename(fid)
#     dirname = os.path.dirname(fid)
#     dirfiles = os.listdir(dirname)
#
#     if basename in dirfiles:
#         return fid
#     else:
#         basename = basename.split(".")[0]
#         files = [f for f in dirfiles if basename in f]
#         if extension is not None:
#             files = [f for f in files if extension in f]
#         if len(files) > 0:
#             return dirname + "/" + files[0]
#         else:
#             print("Error: can't find file")
#
#
# def remove_exts(name, exts):
#     for ext in exts:
#         name = name.replace(ext, "")
#     return name
#
#
# class Dataset_v1(data.Dataset):
#     def __init__(self, seqlist, history=90):
#         self.seqlist = seqlist
#         self.poselist = []
#         self.labellist = []
#         threed_poseloc = '/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/Modified poses/withNeck/'
#         labelloc = '/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/Labels/'
#         labelnames = list(np.load('/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/uniquelabelnames.npy'))
#         for seq in seqlist:
#             threepose = np.load(threed_poseloc + seq + '_pose3d.npy')
#             labels_txt = readtxt(labelloc + seq + '.txt')
#             labels = label2index(labels_txt, labelnames)
#             threepose = threepose[0:len(labels)]
#             for lab in labels:
#                 self.labellist.append(lab)
#             for lab in threepose:
#                 self.poselist.append(lab)
#         print('Read Data')
#
#     def __len__(self):
#         return len(self.poselist)
#
#     def __getitem__(self, index):
#         pose = self.poselist[index]
#         lab = self.labellist[index]
#         return np.double(pose), np.double(np.asarray(lab))
#
#
# class Dataset_v2(data.Dataset):
#     def __init__(self, seqlist, history=90):
#         self.seqlist = seqlist
#         self.poselist = []
#         self.labellist = []
#         self.history = history
#
#         # labelnames = list(np.load('/home/smartlab/Documents/Dataset/UW_IOM/Labels/uniquelabelnames.npy'))
#         # jointfeaturesloc = '/home/smartlab/Documents/Dataset/UW_IOM/Poses/LCRnet/Features3D/node/'
#         # jointfeaturesloc = '/home/smartlab/Documents/Dataset/UW_IOM/Poses/LCRnet/3D Ready/withNeck/'
#         threed_poseloc = '/home/smartlab/Documents/Dataset/UW_IOM/Poses/LCRnet/3D Ready/withNeck/'
#         labelloc = '/home/smartlab/Documents/Dataset/UW_IOM/Labels/tags/'
#         labelnames = list(np.load('/home/smartlab/Documents/Dataset/UW_IOM/Labels/uniquelabelnames.npy'))
#         for seq in seqlist:
#
#             # joint_features = pkl.load(open(jointfeaturesloc + 'node_attr_' + seq + '_pose3d', "rb"))['node_glob_pos'][:,
#             #                  :-1, :]
#             # threepose = pkl.load(open(jointfeaturesloc + 'node_attr_' + seq + '_pose3d', "rb"))[
#             #     'node_glob_pos']  #
#             threepose = np.load(threed_poseloc + seq + '_pose3d.npy')
#             labels_txt = readtxt(labelloc + seq + '.txt')
#             labels = label2index(labels_txt, labelnames)
#             # if seq=='15':
#             #     print(seq)
#             if len(labels) != np.shape(threepose)[0]:
#                 raise Exception('Labels and data dimension does not match in video {}'.format(seq))
#
#             # threepose = threepose[0:len(labels)]
#             x, y = self.truncatewithhistory(threepose, labels)
#             for xx in x:
#                 self.poselist.append(xx)
#             for yy in y:
#                 self.labellist.append(yy)
#         print('Read Data')
#
#     def __len__(self):
#         return len(self.poselist)
#
#     def truncatewithhistory(self, x, y):
#         x_trunk = []
#         y_trunk = []
#         for i in range(0, len(y), self.history):
#             xt = x[i:i + self.history]
#             yt = y[i:i + self.history]
#             if len(yt) == self.history:
#                 x_trunk.append(xt)
#                 y_trunk.append(yt)
#             else:
#                 xt = x[-self.history:]
#                 yt = y[-self.history:]
#                 x_trunk.append(xt)
#                 y_trunk.append(yt)
#
#         return x_trunk, y_trunk
#
#     def __getitem__(self, index):
#         pose = self.poselist[index]
#         lab = self.labellist[index]
#         # print(pose.shape, np.asarray(lab).shape)
#         return np.double(pose), np.double(np.asarray(lab))
#
#
# class Dataset_v3(data.Dataset):
#     """
#     More sampling self.history/2
#     """
#
#     def __init__(self, seqlist, history=90):
#         self.seqlist = seqlist
#         self.poselist = []
#         self.labellist = []
#         self.history = history
#         threed_poseloc = '/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/Modified poses/withNeck/'
#         labelloc = '/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/Labels/'
#         labelnames = list(np.load('/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/uniquelabelnames.npy'))
#         for seq in seqlist:
#             threepose = np.load(threed_poseloc + seq + '_pose3d.npy')
#             labels_txt = readtxt(labelloc + seq + '.txt')
#             labels = label2index(labels_txt, labelnames)
#             threepose = threepose[0:len(labels)]
#             x, y = self.truncatewithhistory(threepose, labels)
#             for xx in x:
#                 self.poselist.append(xx)
#             for yy in y:
#                 self.labellist.append(yy)
#         print('Read Data')
#
#     def __len__(self):
#         return len(self.poselist)
#
#     def truncatewithhistory(self, x, y):
#         x_trunk = []
#         y_trunk = []
#         for i in range(0, len(y), int(self.history / 2)):
#             xt = x[i:i + self.history]
#             yt = y[i:i + self.history]
#             if len(yt) == self.history:
#                 x_trunk.append(xt)
#                 y_trunk.append(yt)
#             else:
#                 xt = x[-self.history:]
#                 yt = y[-self.history:]
#                 x_trunk.append(xt)
#                 y_trunk.append(yt)
#
#         return x_trunk, y_trunk
#
#     def __getitem__(self, index):
#         pose = self.poselist[index]
#         lab = self.labellist[index]
#         # print(pose.shape, np.asarray(lab).shape)
#         return np.double(pose), np.double(np.asarray(lab))

# labelnames = list(np.load('/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/uniquelabelnames.npy'))
# print('done')
# train_split = np.load('/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/TrainValSplits/train.npy')
# val_split = np.load('/home/anarayanan/LCRNet_v2.0/UW/Behnoosh/UW IOM Dataset/TrainValSplits/val.npy')
# criterion = nn.CrossEntropyLoss()
# criterion.cuda()
# params_train = {'batch_size': 128, 'shuffle': True}
# training_set = Dataset_v1(train_split)
# training_generator = data.DataLoader(training_set, **params_train)
# params_val = {'batch_size': 128, 'shuffle': False}
# val_set = Dataset_v1(val_split)
# val_generator = data.DataLoader(val_set, **params_val)
# for local_im, local_labels in val_generator:
#     pass
