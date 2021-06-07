from torch import optim

from vis.plotRibbons import plot_reba, plot_sequence
from val.validate_model_UW import *
from models.model_MT_UW import *
# from vis.plotCM import plot_confusion_matrix
# from util.utils import *
from scipy.stats import spearmanr
from vis.plotCM import *
from util.utils import *
import yaml
try:
    with open('./config_files/config_UW_data.yml', 'r') as file:
        config_data = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')
try:
    with open('./config_files/config_UW_class.yml', 'r') as file:
        config_exp = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')

base_data_dir = config_data['base_data_dir']
num_class = config_data['NUMBER_OF_CLASSES']
val_split = np.load(base_data_dir+config_data['val_dir'])
threed_poseloc = base_data_dir+config_data['threed_poseloc']
labelloc = base_data_dir+config_data['label_dir']
reba_scores_loc = base_data_dir+config_data['reba_scores_loc']
labelnames = base_data_dir+config_data['labelnames']
classes = [
  'walking', 'stand_reach_top', 'box_stand_pick_up_top', 'box_stand_place_mid', 'standing', 'rod_stand_pick_up_top',
  'rod_stand_place_mid', 'box_stand_pick_up_mid', 'rod_stand_pick_up_mid', 'none_bend_none_none',
  'rod_bend_pick_up_low',
  'box_bend_place_low', 'rod_bend_place_low', 'box_stand_place_top', 'rod_stand_place_top', 'box_bend_pick_up_low',
  'box_walk_hold']

# base
EXP_name = 'base'
CHECKPOINT_PATH =  './outputs/UW/checkpoints/0.001_SmoothREBA_gcnEdtcnREBA_tanh_MSEL1_CrossEntropy.pt'
checkpoint = torch.load(CHECKPOINT_PATH)
n_nodes = [50, 50, 50, 50]
model = gcnEdtcnREBA_tanh(hidden=n_nodes, kernel_size=4).cuda()
model.load_state_dict(checkpoint)

listpred_for_CM, labellist, flat_listpred_for_CM, label_gt, label_pred, reba_gt, reba_pred, result, coef_list, EDIT, OVERLAP_F1, MSE = eval(model)

# Results
res = []
for k in result.keys():
    res.append(100*result[k])

print('MSE: \n', MSE)
print('mean(MSE): \n', np.mean(MSE))
print('std(MSE): \n', np.std(MSE))
print('Corr: \n', coef_list)
print('mean(Corr): \n', np.mean(coef_list))
print('std(Corr): \n', np.std(coef_list))
print('meanEdit: ' + str(round(np.mean(EDIT), 4)))
print('stdEdit: ' + str(round(np.std(EDIT), 4)))
print('meanF1: ' + str(round(np.mean(OVERLAP_F1), 4)))
print('stdF1: ' + str(round(np.std(OVERLAP_F1), 4)))
print('AP_Scores: \n', res)
print('mean_AP_Score: ' + str(round(np.mean(res), 4)))
print('std_AP_Score: ' + str(round(np.std(res), 4)))
#
#
# # Plots
# plot_sequence(listpred_for_CM, labellist, classes, saving_dir='UW_seq_'+EXP_name)
# conf_mat = confusion_matrix(label_gt, flat_listpred_for_CM)
# np.save('CM_UW_'+EXP_name, conf_mat)
# plot_confusion_matrix(conf_mat, classes,'CM_UW_'+EXP_name)
# plot_reba(reba_pred, reba_gt, labellist, classes,data='UW', saving_dir='UW_REBA_'+EXP_name)

# %% Emb
# EXP_name = 'emb'
# # emb
# CHECKPOINT_PATH = './outputs/UW/checkpoints/0.0001_SmoothREBA_3_gcnEdtcn_emb_final_MSEL1_CrossEntropy.pt'
# # CHECKPOINT_PATH = './results_WACV2021/UW/checkpoints/0.001_GCN_True_EDTCN_newREBA_H50_MT_Cross_MSE_L1_emb.pt'
# checkpoint = torch.load(CHECKPOINT_PATH)
# n_nodes = [50, 50, 50, 50]
# model = gcnEdtcnREBA_emb_final(hidden=n_nodes, kernel_size=4).cuda()
#
# model.load_state_dict(checkpoint)
# listpred_for_CM, labellist, flat_listpred_for_CM, label_gt, label_pred, reba_gt, reba_pred, result, coef_list, EDIT, OVERLAP_F1, MSE = eval(model)
# # Results
# res = []
# for k in result.keys():
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
# plot_sequence(listpred_for_CM, labellist, classes, saving_dir='UW_seq_'+EXP_name)
# conf_mat = confusion_matrix(label_gt, flat_listpred_for_CM)
# np.save('CM_UW_'+EXP_name, conf_mat)
# plot_confusion_matrix(conf_mat, classes,'CM_UW_'+EXP_name)
# plot_reba(reba_pred, reba_gt, labellist, classes,data='UW', saving_dir='UW_REBA_'+EXP_name)

# %% Compare Confusion Matrices
CM_dir_base = './CM_UW_base.npy'
CM_dir_emb = './CM_UW_emb.npy'

conf_mat_base = np.load(CM_dir_base)
CM_base = plot_confusion_matrix(conf_mat_base, classes, 'UW_conf_mat_base'+'.eps') #'UW_conf_mat_base'+'.png'

conf_mat_emb = np.load(CM_dir_emb)
CM_emb = plot_confusion_matrix(conf_mat_emb, classes, 'UW_conf_mat_emb'+'.eps')#'UW_conf_mat_emb'+'.png'

CM = CM_base - CM_emb
plot_confusion_matrix(CM, classes, 'UW_diff'+'.eps', normalize=False,cmap=plt.cm.PiYG)
