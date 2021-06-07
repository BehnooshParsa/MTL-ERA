import matplotlib.pylab as plt
import matplotlib as mlt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.signal import butter, lfilter, freqz
# classes_list = ( 'box_bend_pick-up_low', 'box_bend_place_low', 'box_stand_pick-up_mid', 'box_stand_pick-up_top', 'box_stand_place_mid', 'box_stand_place_top', 'box_walk_hold_none', 'bend', 'stand', 'stand_reach_top', 'walk', 'rod_bend_pick-up_low', 'rod_bend_place_low', 'rod_stand_pick-up_mid', 'rod_stand_pick-up_top', 'rod_stand_place_mid', 'rod_stand_place_top')
col_list = ["viridian", "reddish orange", "dark magenta", "neon blue", "light grey",  "gunmetal", "hot pink", "baby blue", "green blue", "baby pink", "light yellow", "magenta",
                "twilight blue", "neon purple", "cool blue", "blue green", "brown", "magenta", "dark blue", "green","baby purple"]
REBA_list = list(str(i) for i in range(0,15))
# %% visualizing the segmentation ribbons
def imshow_(fig, ax, x, **kwargs):
    n_classes = kwargs['n_classes']
    col_list_palette = sns.xkcd_palette(col_list[0:n_classes])
    CustomCmap = mlt.colors.ListedColormap(col_list_palette)
    if x.ndim == 2:
        ax.imshow(x, interpolation="nearest", cmap=CustomCmap, aspect=50)
    elif x.ndim == 1:
        ax.imshow(x[:, None].T, interpolation="nearest", cmap=CustomCmap, aspect=50)
        ax.set_yticks([])
    ax.axis("tight")



def plot_sequence(P_test, y_test, classes_list, saving_dir=None):
    fig, axs = plt.subplots(len(y_test), 1, figsize=(20, 15))
    n_classes = len(classes_list)
    for i in range(len(y_test)):
        P_tmp = np.vstack([np.expand_dims(y_test[i], axis=0), np.expand_dims(P_test[i], axis=0)])
        # plt.subplot(len(y_test), 1, i + 1)
        imshow_(fig,axs[i],P_tmp, n_classes=n_classes, vmin=0, vmax=1)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        acc = np.mean(y_test[i] == P_test[i]) * 100
        axs[i].set_ylabel("Acc: {:.01f}%".format(acc))

    classes = np.arange(0, n_classes)
    values = np.unique(classes.ravel())
    patches = [
        mpatches.Patch(color=mlt.colors.to_rgba(sns.xkcd_rgb[col_list[i]]), label=classes_list[values[i]]) for i
        in range(len(values))]

    axs[0].legend(handles=patches, loc=2, bbox_to_anchor=(1, 1.))

    if saving_dir is not None:
        plt.savefig(os.path.join(saving_dir + '.png'))
        # fig.set_rasterized(True) # messes with eps
        plt.savefig(os.path.join(saving_dir + '.eps'), format="eps")
    plt.show()


def plot_reba(P_test, y_test,labels, classes_list, saving_dir=None, data='UW'):
    fig, axs = plt.subplots(len(y_test), 1, figsize=(20, 15))
    for i in range(len(y_test)):
        n = y_test[i].shape[0]
        t = np.arange(0, y_test[i].shape[0])
        reba_to_plot = P_test[i] #np.round(P_test[i])
        axs[i].plot(t, y_test[i], label='Ground Truth',linewidth=3, color=mlt.colors.to_rgba(sns.xkcd_rgb['light grey']))
        index_ = [[0, int(labels[i][0])]]

        for j in range(1,n):
            if labels[i][j]-labels[i][j-1]!= 0:
                index_.append([j, int(labels[i][j])])

        for k in range(1, len(index_)):
            if data == 'TUM':
                if index_[k-1][1]>13:
                    dd = index_[k-1][1]-1
                else:
                    dd = index_[k-1][1]
            else:
                dd = index_[k - 1][1]
            axs[i].plot(t[index_[k-1][0]:index_[k][0]], reba_to_plot[index_[k-1][0]:index_[k][0]], linewidth=3,
                        color=mlt.colors.to_rgba(sns.xkcd_rgb[col_list[dd]]), label=classes_list[dd])

        mse = mean_squared_error(P_test[i], y_test[i])
        axs[i].set_xticks([])
        # axs[i].set_yticks([])
        axs[i].set_ylabel("MSE: {:.01f}%".format(mse))
        axs[i].axis("tight")

    if saving_dir is not None:
        plt.savefig(os.path.join(saving_dir+'_REBA' + '.png'))
        # fig.set_rasterized(True) # messes with eps
        plt.savefig(os.path.join(saving_dir+'_REBA' + '.eps'), format="eps")
    plt.show()

    # def plot_reba(P_test, y_test,labels, n_classes, saving_dir=None):
    #     fig, axs = plt.subplots(len(y_test), 1, figsize=(20, 15))
    #     for i in range(len(y_test)):
    #         n = y_test[i].shape[0]
    #         t = np.arange(0, y_test[i].shape[0])
    #         reba_to_plot = P_test[i] #np.round(P_test[i])
    #         axs[i].plot(t, y_test[i], label='Ground Truth',linewidth=3, color=mlt.colors.to_rgba(sns.xkcd_rgb['light grey']))
    #         index_ = [[0, labels[i][0]]]
    #         for j in range(1,n):
    #             if labels[i][j]-labels[i][j-1]!= 0:
    #                 index_.append([j, labels[i][j]])
    #
    #         for k in range(1, len(index_)):
    #             axs[i].plot(t[index_[k-1][0]:index_[k][0]], reba_to_plot[index_[k-1][0]:index_[k][0]], linewidth=3,
    #                         color=mlt.colors.to_rgba(sns.xkcd_rgb[col_list[index_[k-1][1]]]), label=classes_list[index_[k-1][1]])
    #
    #         mse = mean_squared_error(P_test[i], y_test[i])
    #         axs[i].set_xticks([])
    #         # axs[i].set_yticks([])
    #         axs[i].set_ylabel("MSE: {:.01f}%".format(mse))
    #         axs[i].axis("tight")
    #     plt.show()
    ## Plot bars
    # fig, axs = plt.subplots(len(y_test), 1, figsize=(20, 15))
    # for i in range(len(y_test)):
    #     n = y_test[i].shape[0]
    #     t = np.arange(0, y_test[i].shape[0])
    #     reba_to_plot = np.round(P_test[i])
    #     axs[i].fill_between(t, y_test[i],0, color=mlt.colors.to_rgba(sns.xkcd_rgb['black']))
    #     index_ = [[0, labels[i][0]]]
    #     for j in range(1,n):
    #         if labels[i][j]-labels[i][j-1]!= 0:
    #             index_.append([j, labels[i][j]])
    #
    #     for k in range(1, len(index_)):
    #         axs[i].fill_between(t[index_[k-1][0]:index_[k][0]], reba_to_plot[index_[k-1][0]:index_[k][0]],0 , linewidth=2,
    #                     color=mlt.colors.to_rgba(sns.xkcd_rgb[col_list[index_[k-1][1]]]), label=classes_list[index_[k-1][1]])
    #
    #     mse = mean_squared_error(np.round(P_test[i]), y_test[i])
    #     axs[i].set_xticks([])
    #     axs[i].set_yticks([])
    #     axs[i].set_ylabel("MSE: {:.01f}%".format(mse))
    # plt.show()
    # classes = np.arange(0, n_classes)
    # values = np.unique(classes.ravel())
    # patches = [
    #     mpatches.Patch(color=mlt.colors.to_rgba(sns.xkcd_rgb[col_list[i]]), label=classes_list[values[i]]) for i
    #     in range(len(values))]
    # axs[0].legend(handles=patches, loc=2, bbox_to_anchor=(1, 1.))
