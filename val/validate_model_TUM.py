import torch
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
import math
from util.metrics import *
from torch.autograd import Variable
from util.utils import mask_data, unmask
from vis.plotCM import compute_confusion_matrix
from termcolor import colored
import yaml

try:
    with open('./config_files/config_TUM_data.yml', 'r') as file:
        config_data = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')
try:
    with open('./config_files/config_TUM_exp.yml', 'r') as file:
        config_exp = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')


base_data_dir = config_data['base_data_dir']

num_class = config_data['NUMBER_OF_CLASSES']
val_split = np.load(base_data_dir + config_data['val_dir'])
threed_poseloc = base_data_dir + config_data['threed_poseloc']
labelloc = base_data_dir + config_data['label_dir']
reba_scores_loc = base_data_dir + config_data['reba_scores_loc']

def eval(model):
    model.eval()

    labellist = []
    predlist = []
    EDIT = []
    OVERLAP_F1 = []
    MSE = []
    reba_pre_list = []
    reba_gt_list = []
    listpred_for_CM = []
    coef_list = []
    max_len = 2384
    with torch.no_grad():
        for seq in val_split:
            threepose = np.load(threed_poseloc + seq + '.npy').astype(float)
            labels_gt = np.loadtxt(labelloc + seq + '.txt')
            reba_gt = np.loadtxt(reba_scores_loc + seq + '.txt')
            threepose, _, _, mask = mask_data([threepose], [labels_gt], [reba_gt], max_len, mask_value=-1)
            x = Variable(torch.Tensor(threepose)).float().cuda()
            score, reba_pre = model(x)
            score = unmask(score.cpu().numpy(), mask)
            reba_pre = unmask(reba_pre.cpu().numpy(), mask)

            scorelist = [score]
            rebalist = [reba_pre]
            ll = np.asarray(labels_gt)
            lp = np.asarray([item for sublist in scorelist for item in sublist])
            llp = np.argmax(lp.squeeze(), axis=1)
            reba_pre = np.asarray([item for sublist in rebalist for item in sublist]).squeeze()
            coef, p = spearmanr(reba_pre, reba_gt)
            coef_list.append(100 * coef)
            print('Spearmans correlation coefficient: %.3f' % coef)
            # interpret the significance
            alpha = 0.05
            if p > alpha:
                print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
            else:
                print('Samples are correlated (reject H0) p=%.3f' % p)
            MSE.append(((reba_pre - reba_gt) ** 2).mean(axis=0))
            labellist.append(ll)
            predlist.append(lp.squeeze())
            listpred_for_CM.append(llp)
            reba_pre_list.append(reba_pre)
            reba_gt_list.append(reba_gt)
            EDIT.append(edit_score(llp, ll))
            OVERLAP_F1.append(overlap_f1(llp, ll, n_classes=num_class))

        flat_listlabel = [item for sublist in labellist for item in sublist]
        flat_listpred = [item for sublist in predlist for item in sublist] # scores
        flat_listpred_for_CM = [item for sublist in listpred_for_CM for item in sublist] # label class
        result = compute_class_ap(np.asarray(flat_listpred), np.asarray(flat_listlabel))
        return listpred_for_CM, labellist, flat_listpred_for_CM,  flat_listlabel, flat_listpred, reba_gt_list, reba_pre_list, result, coef_list, EDIT, OVERLAP_F1, MSE


def compute_class_ap(pred_list, label_list):
    result = OrderedDict()
    for cls in range(0, num_class):
        nmn = average_precision_score((label_list == cls).astype(np.uint8), pred_list[:, cls])
        if math.isnan(nmn):
            nmn = 0
        result[cls] = nmn
    return result


def val(generator, model, MT_losses):
    model.eval()
    with torch.no_grad():
        losses = 0.0
        losses_reg = 0.0
        losses_class = 0.0
        for local_im, local_labels, reba_gt in generator:
            local_im, local_labels, reba_gt = local_im.float().cuda(), local_labels.long().cuda(), reba_gt.float().cuda()

            loss_class, loss_reg, loss = MT_losses(local_im, [local_labels, reba_gt])
            losses = losses + loss.cpu().data.numpy()
            losses_reg = losses_reg + loss_reg.cpu().data.numpy()
            losses_class = losses_class + loss_class.cpu().data.numpy()
    model.train()
    return losses, losses_reg, losses_class


def bestval(generator, max_len, model, n_class):
    model.eval()
    losses = 0.0

    labellist = []
    predlist = []
    # flat_listpred=[]
    # flat_listlabel=[]
    EDIT = []
    OVERLAP_F1 = []
    MSE = []
    listpred_for_CM = []
    spearmanr_list = []
    with torch.no_grad():
        for seq in val_split:
            threepose = np.load(threed_poseloc + seq + '.npy').astype(float)
            # labels_txt = readtxt(labelloc + seq + '.txt')
            labels_gt =  np.loadtxt(labelloc + seq + '.txt')  #label2index(labels_txt, labelnames)
            reba_gt = np.loadtxt(reba_scores_loc + seq + '.txt')
            threepose, _, _, mask = mask_data([threepose], [labels_gt], [reba_gt], max_len, mask_value=-1)
            x = Variable(torch.Tensor(threepose)).float().cuda()
            score, reba_pre = model(x)
            score = unmask(score.cpu().numpy(), mask)
            reba_pre = unmask(reba_pre.cpu().numpy(), mask)
            scorelist = [score]
            rebalist = [reba_pre]

            ll = np.asarray(labels_gt)
            lp = np.asarray([item for sublist in scorelist for item in sublist])
            llp = np.argmax(lp.squeeze(), axis=1)
            EDIT.append(edit_score(llp, ll))
            OVERLAP_F1.append(overlap_f1(llp, ll, n_classes=num_class))
            reba_pre = np.asarray([item for sublist in rebalist for item in sublist]).squeeze()
            MSE.append(((reba_pre - reba_gt) ** 2).mean(axis=0))
            coef, p = spearmanr(reba_pre, reba_gt)
            spearmanr_list.append([coef, p])
            labellist.append(ll)
            predlist.append(lp.squeeze())
            listpred_for_CM.append(llp)
        flat_listlabel = [item for sublist in labellist for item in sublist]
        flat_listpred = [item for sublist in predlist for item in sublist]
        # flat_listpred_for_CM = [item for sublist in listpred_for_CM for item in sublist]
        result = compute_class_ap(np.asarray(flat_listpred), np.asarray(flat_listlabel))
    conf_mat, class_accuracy = compute_confusion_matrix(n_class, generator, model)
    # conf_mat2 = confusion_matrix(flat_listlabel, flat_listpred_for_CM)
    model.train()
    return result, EDIT, OVERLAP_F1, MSE, conf_mat, class_accuracy, spearmanr_list


def check(tensor_):
    n = tensor_.shape[0]
    d = 0
    for item in tensor_:
        if item == 0:
            d += 1
    return d == n


def show_results(lr, epoch, vallosses, OUTPUTFILE, CM_dir, generator_val, max_len, model, mt_loss, n_classes):
    result, ed, f1, MSE, conf_mat, class_accuracy, spearmanr_list = bestval(generator_val, max_len, model, n_classes)
    print(colored('meanEdit: ' + str(round(np.mean(ed), 4)), 'cyan'))
    print(colored('meanF1: ' + str(round(np.mean(f1), 4)), 'cyan'))
    print(colored('meanMSE: ' + str(round(np.mean(MSE), 4)), 'cyan'))
    print(colored('loss weights: ' + str(mt_loss.eta.data), 'cyan'))
    # print('class_accuracy: ', class_accuracy)
    # print(conf_mat)
    # plt.figure(figsize=(NUMBER_OF_CLASSES, NUMBER_OF_CLASSES))
    # plot_confusion_matrix(conf_mat, classes, CM_dir+'confusion'+OUTPUTFILE[:-4])
    # plt.savefig(CM_dir + 'confusion' + OUTPUTFILE[:-4])
    # print(class_accuracy)
    np.save(CM_dir + str(lr) + '_confusion_' + OUTPUTFILE[:-4], conf_mat)
    meanresult = []
    meanresult.append(['lr', lr])
    meanresult.append(['epoch', epoch])
    meanresult.append(['vallosses', vallosses])
    for k in result.keys():
        meanresult.append(result[k])
    # meanresult.append(['meanresult',np.mean(meanresult[3:])])
    # meanresult.append(['meanEdit',np.mean(ed)])
    # meanresult.append(['meanF1',np.mean(f1)])
    output_file = open(config_exp['log_dir'] + 'maps_' + OUTPUTFILE, 'a')
    output_file.write('\n------------------\n')
    output_file.write('Saved Epoch: ' + str(epoch) + ' Valloss: ' + str(round(vallosses, 4)) + '\n')
    output_file.write('mean_AP_Score: ' + str(round(np.mean(meanresult[3:]), 4)) + '\n')
    output_file.write('std_AP_Score: ' + str(round(np.std(meanresult[3:]), 4)) + '\n')
    output_file.write('meanEdit: ' + str(round(np.mean(ed), 4)) + '\n')
    output_file.write('stdEdit: ' + str(round(np.std(ed), 4)) + '\n')
    output_file.write('meanF1: ' + str(round(np.mean(f1), 4)) + '\n')
    output_file.write('stdF1: ' + str(round(np.std(f1), 4)) + '\n')
    output_file.write('meanMSE: ' + str(round(np.mean(MSE), 4)) + '\n')
    output_file.write('stdMSE: ' + str(round(np.std(MSE), 4)) + '\n')
    output_file.write('spearmanr (coef, p): ' + str(spearmanr_list) + '\n')
    output_file.write('loss weights: ' + str(mt_loss.eta.data) + '\n')
    output_file.write('class_accuracy: ')
    for i, item in enumerate(list(class_accuracy)):
        # output_file.writelines(str(i) + ": ")
        output_file.writelines(str(round(item, 4)) + " ")
    output_file.write('\n')
    output_file.write('class_AP_Score: ')
    for i, item in enumerate(meanresult[3:]):
        # output_file.writelines(str(i) + ": ")
        output_file.writelines(str(round(item, 4)) + " ")
    output_file.write('\n')
    # output_file.write(str(meanresult) + '\n')
    output_file.close()
    return mt_loss.eta.data


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, mt_losses, dir_out, outputfile, CM_dir, lr, epoch, generator_val, temporal_len,
                 n_classes):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, dir_out, str(lr) + '_' + outputfile)
        elif math.isnan(score):
            self.early_stop = True
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, dir_out, str(lr) + '_' + outputfile)
            self.counter = 0
            eta = show_results(lr, epoch, val_loss, outputfile, CM_dir, generator_val, temporal_len, model, mt_losses,
                               n_classes)
            if check(eta) == True:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, dir_out, outputfile):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(colored(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',
                          'cyan'))
        torch.save(model.state_dict(), dir_out + outputfile.replace('.txt', '.pt'))
        self.val_loss_min = val_loss

#
# def val_history(generator, model, MT_losses):
#     model.eval()
#     with torch.no_grad():
#         losses = 0.0
#         losses_reg = 0.0
#         losses_class = 0.0
#         for local_im, local_labels, reba_gt in generator:
#             local_im, local_labels, reba_gt = local_im.float().cuda(), local_labels.long().cuda(), reba_gt.float().cuda()
#             # score, reba_pre = model(local_im)
#             # loss_class = criterion_class(score.view(-1, 17), local_labels.view(-1))
#             # loss_reg = criterion_reg[0](reba_pre.view(-1), reba_gt.view(-1)) + criterion_reg[1](reba_pre.view(-1),
#             #                                                                                     reba_gt.view(-1))
#             # loss = loss_class + loss_reg
#             loss_class, loss_reg, loss = MT_losses(local_im, [local_labels, reba_gt])
#             losses = losses + loss.cpu().data.numpy()
#             losses_reg = losses_reg + loss_reg.cpu().data.numpy()
#             losses_class = losses_class + loss_class.cpu().data.numpy()
#     model.train()
#     return losses, losses_reg, losses_class
#
#
# def bestval_history(generator, model, HISTORY, n_class):
#     model.eval()
#     losses = 0.0
#     val_split = np.load('/home/smartlab/Documents/Dataset/UW_IOM/EDTCN/TrainValSplits/val.npy')
#     threed_poseloc = '/home/smartlab/Documents/Dataset/UW_IOM/Poses/LCRnet/3D Ready/withNeck/'
#     labelloc = '/home/smartlab/Documents/Dataset/UW_IOM/Labels/tags/'
#     labelnames = list(np.load('/home/smartlab/Documents/Dataset/UW_IOM/Labels/uniquelabelnames.npy'))
#     reba_scores_loc = '/home/smartlab/Documents/Dataset/UW_IOM/RebaScoreFrameWise/'
#     labellist = []
#     predlist = []
#     # flat_listpred=[]
#     # flat_listlabel=[]
#     EDIT = []
#     OVERLAP_F1 = []
#     MSE = []
#     listpred_for_CM = []
#     with torch.no_grad():
#         for seq in val_split:
#             threepose = np.load(threed_poseloc + seq + '_pose3d.npy')
#             labels_txt = readtxt(labelloc + seq + '.txt')
#             labels = label2index(labels_txt, labelnames)
#             reba_gt = np.loadtxt(reba_scores_loc + seq + '.txt')
#             threepose = threepose[0:len(labels)]
#             # for local_im, local_labels in generator:
#             scorelist = []
#             rebalist = []
#             for i in range(0, threepose.shape[0], HISTORY):
#                 x = Variable(torch.Tensor(threepose[i:i + HISTORY])).float().cuda()
#                 if x.shape[0] != 80:
#                     xx = Variable(torch.Tensor(threepose[-HISTORY:])).float().cuda()
#                     score, reba_pre = model(xx.unsqueeze(0))
#                     scorelist.append(score[0, -x.shape[0]:].cpu().data.numpy())
#                     rebalist.append(reba_pre[0, -x.shape[0]:].cpu().data.numpy())
#                 else:
#                     score, reba_pre = model(x.unsqueeze(0))
#                     scorelist.append(score[0].cpu().data.numpy())
#                     rebalist.append(reba_pre[0].cpu().data.numpy())
#             ll = np.array(labels)
#             lp = np.asarray([item for sublist in scorelist for item in sublist])
#             llp = np.argmax(lp, axis=1)
#             EDIT.append(edit_score(llp, ll))
#             OVERLAP_F1.append(overlap_f1(llp, ll, n_classes=17))
#             reba_pre = np.asarray([item for sublist in rebalist for item in sublist]).squeeze()
#             MSE.append(((reba_pre - reba_gt) ** 2).mean(axis=0))
#             labellist.append(ll)
#             predlist.append(lp)
#             listpred_for_CM.append(llp)
#         flat_listlabel = [item for sublist in labellist for item in sublist]
#         flat_listpred = [item for sublist in predlist for item in sublist]
#         # flat_listpred_for_CM = [item for sublist in listpred_for_CM for item in sublist]
#         result = compute_class_ap(np.asarray(flat_listpred), np.asarray(flat_listlabel))
#     conf_mat, class_accuracy = compute_confusion_matrix(n_class, generator, model)
#     # conf_mat2 = confusion_matrix(flat_listlabel, flat_listpred_for_CM)
#     model.train()
#     return result, EDIT, OVERLAP_F1, MSE, conf_mat, class_accuracy
#
#
# def show_results_history(lr, epoch, vallosses, outputfile, CM_dir, generator_val, history, model, n_classes):
#     result, ed, f1, MSE, conf_mat, class_accuracy = bestval_history(generator_val, model, history,
#                                                                     n_classes)
#     print(colored('meanEdit: ' + str(round(np.mean(ed), 4)), 'cyan'))
#     print(colored('meanF1: ' + str(round(np.mean(f1), 4)), 'cyan'))
#     print(colored('meanMSE: ' + str(round(np.mean(MSE), 4)), 'cyan'))
#     # print('class_accuracy: ', class_accuracy)
#     # plot_confusion_matrix(conf_mat, classes, CM_dir+'confusion'+OUTPUTFILE[:-4])
#     # print(class_accuracy)
#     np.save(CM_dir + str(lr) + '_confusion_' + outputfile[:-4], conf_mat)
#     meanresult = []
#     meanresult.append(['lr', lr])
#     meanresult.append(['epoch', epoch])
#     meanresult.append(['vallosses', vallosses])
#     for k in result.keys():
#         meanresult.append(result[k])
#     # meanresult.append(['meanresult',np.mean(meanresult[3:])])
#     # meanresult.append(['meanEdit',np.mean(ed)])
#     # meanresult.append(['meanF1',np.mean(f1)])
#     output_file = open('./results_WACV2021/maps_' + outputfile, 'a')
#     output_file.write('\n------------------\n')
#     output_file.write('Saved Epoch: ' + str(epoch) + ' Valloss: ' + str(round(vallosses, 4)) + '\n')
#     output_file.write('mean_AP_Score: ' + str(round(np.mean(meanresult[3:]), 4)) + '\n')
#     output_file.write('meanEdit: ' + str(round(np.mean(ed), 4)) + '\n')
#     output_file.write('meanF1: ' + str(round(np.mean(f1), 4)) + '\n')
#     output_file.write('meanMSE: ' + str(round(np.mean(MSE), 4)) + '\n')
#     output_file.write('class_accuracy: ')
#     for i, item in enumerate(list(class_accuracy)):
#         # output_file.writelines(str(i) + ": ")
#         output_file.writelines(str(round(item, 4)) + " ")
#     output_file.write('\n')
#     output_file.write('class_AP_Score: ')
#     for i, item in enumerate(meanresult[3:]):
#         # output_file.writelines(str(i) + ": ")
#         output_file.writelines(str(round(item, 4)) + " ")
#     output_file.write('\n')
#     # output_file.write(str(meanresult) + '\n')
#     output_file.close()
