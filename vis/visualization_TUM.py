from torch import optim

from models.model_MT_TUM import gcnEdtcnREBA_emb
from vis.plotRibbons import plot_reba, plot_sequence
from val.validate_model_TUM import *
from models.model_MT_TUM import *
from vis.plotCM import *
from util.utils import *
import yaml
# Note that TUM technically has 20 classes. Class 13 is missing but since the labels go to 20 we had to set class number to 21 but we are taking care of it when plotting CM.
try:
    with open('./config_files/config_TUM_data.yml', 'r') as file:
        config_data = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')
try:
    with open('./config_files/config_TUM_class.yml', 'r') as file:
        config_exp = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')

base_data_dir = config_data['base_data_dir']
num_class = config_data['NUMBER_OF_CLASSES']
val_split = np.load(base_data_dir+config_data['val_dir'])
threed_poseloc = base_data_dir+config_data['threed_poseloc']
labelloc = base_data_dir+config_data['label_dir']
reba_scores_loc = base_data_dir+config_data['reba_scores_loc']
classes = ['close,cabinet', 'close,drawer', 'open,cabinet', 'open,drawer', 'pick-up,cabinet', 'pick-up,drawer', 'pick-up,hold-both-hand', 'pick-up,hold-one-hand','place,hold-both-hand','place,hold-one-hand',
         'reach,cabinet', 'reach,drawer','reach,not-hold', 'stand,not-hold', 'twist,hold-both-hand', 'twist,hold-one-hand', 'twist,not-hold', 'walk,hold-both-hand', 'walk,hold-one-hand', 'walk,not-hold']


# base
# EXP_name = 'base'
# CHECKPOINT_PATH = './outputs/TUM/checkpoints/0.0001_SmoothREBA_gcnEdtcnREBA_tanh_MSEL1_CrossEntropy.pt'
# checkpoint = torch.load(CHECKPOINT_PATH)
# n_nodes = [50, 50, 50, 50]
# model = gcnEdtcnREBA_tanh(hidden=n_nodes, kernel_size=4).cuda()
# model.load_state_dict(checkpoint)
#
# listpred_for_CM, labellist, flat_listpred_for_CM, label_gt, label_pred, reba_gt, reba_pred, result, coef_list, EDIT, OVERLAP_F1, MSE = eval(model)
# # # Results
# res = []
# for k in result.keys():
#     if k==13:
#         continue
#     res.append(100*result[k])
#
# print('MSE: \n', MSE)
# print('mean(MSE): \n', np.mean(MSE))
# print('std(MSE): \n', np.std(MSE))
# print('Corr: \n', coef_list)
# print('mean(Corr): \n', np.mean(coef_list))
# print('std(Corr): \n', np.std(coef_list))
# print('meanEdit: ' + str(round(np.mean(EDIT), 4)))
# print('stdEdit: ' + str(round(np.std(EDIT), 4)))
# print('meanF1: ' + str(round(np.mean(OVERLAP_F1), 4)))
# print('stdF1: ' + str(round(np.std(OVERLAP_F1), 4)))
# print('AP_Scores: \n', res)
# print('mean_AP_Score: ' + str(round(np.mean(res), 4)))
# print('std_AP_Score: ' + str(round(np.std(res), 4)))
#
#
# # Plots
# plot_sequence(listpred_for_CM, labellist, classes, saving_dir='TUM_seq_'+EXP_name) #
# conf_mat = confusion_matrix(label_gt, flat_listpred_for_CM)
# np.save('CM_TUM_'+EXP_name, conf_mat)
# plot_confusion_matrix(conf_mat, classes,'CM_TUM_'+EXP_name)
# plot_reba(reba_pred, reba_gt, labellist, classes,data='TUM', saving_dir='TUM_REBA_'+EXP_name) #
# %% Emb

# EXP_name = 'emb'
# CHECKPOINT_PATH = './outputs/TUM/checkpoints/0.0005_SmoothREBA_gcnEdtcn_emb_final_MSEL1_CrossEntropy.pt'
# checkpoint = torch.load(CHECKPOINT_PATH)
# n_nodes = [50, 50, 50, 50]
# model = gcnEdtcnREBA_emb_final(hidden=n_nodes, kernel_size=4).cuda()
# model.load_state_dict(checkpoint)
#
# listpred_for_CM, labellist, flat_listpred_for_CM, label_gt, label_pred, reba_gt, reba_pred, result, coef_list, EDIT, OVERLAP_F1, MSE = eval(model)
# Results
# res = []
# for k in result.keys():
#     if k==13:
#         continue
#     res.append(100*result[k])
#
# print('MSE: \n', MSE)
# print('mean(MSE): \n', np.mean(MSE))
# print('std(MSE): \n', np.std(MSE))
# print('Corr: \n', coef_list)
# print('mean(Corr): \n', np.mean(coef_list))
# print('std(Corr): \n', np.std(coef_list))
# print('meanEdit: ' + str(round(np.mean(EDIT), 4)))
# print('stdEdit: ' + str(round(np.std(EDIT), 4)))
# print('meanF1: ' + str(round(np.mean(OVERLAP_F1), 4)))
# print('stdF1: ' + str(round(np.std(OVERLAP_F1), 4)))
# print('AP_Scores: \n', res)
# print('mean_AP_Score: ' + str(round(np.mean(res), 4)))
# print('std_AP_Score: ' + str(round(np.std(res), 4)))
#
#
# # Plots
# plot_sequence(listpred_for_CM, labellist, classes, saving_dir='TUM_seq_'+EXP_name) #
# conf_mat = confusion_matrix(label_gt, flat_listpred_for_CM)
# np.save('CM_TUM_'+EXP_name, conf_mat)
# plot_confusion_matrix(conf_mat, classes,'CM_TUM_'+EXP_name)
# plot_reba(reba_pred, reba_gt, labellist, classes,data='TUM', saving_dir='TUM_REBA_'+EXP_name) #

# %% Compare Confusion Matrices
CM_dir_base = './CM_TUM_base.npy' #'outputs/TUM/plots/Confusion_Matrix/0.0001_confusion_SmoothREBA_gcnEdtcnREBA_tanh_MSEL1_CrossEntropy.npy'
CM_dir_emb = './CM_TUM_emb.npy' #'outputs/TUM/plots/Confusion_Matrix/0.0001_confusion_SmoothREBA_gcnEdtcn_emb_final_MSEL1_CrossEntropy.npy''./results_WACV2021/TUM/Confusion_Matrix/0.0001_confusion_gcnEdtcnREBA_emb2_tanh_MSEL1_CrossEntropy.npy' #


conf_mat_base = np.load(CM_dir_base)
CM_base = plot_confusion_matrix(conf_mat_base, classes, 'TUM_conf_mat_base'+'.eps')

conf_mat_emb = np.load(CM_dir_emb)
CM_emb = plot_confusion_matrix(conf_mat_emb, classes, 'TUM_conf_mat_emb'+'.eps')

CM = -CM_emb+CM_base
plot_confusion_matrix(CM, classes, '_TUM_diff'+'.eps', normalize=False, cmap=plt.cm.PiYG)