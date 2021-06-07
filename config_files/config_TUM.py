import os
import yaml
from termcolor import colored
from torch import nn

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

if config_exp['HISTORY'] == 'None':
    HISTORY = None
    print(colored('Training on the whole sequence', 'blue'))
else:
    HISTORY = config_exp['HISTORY']
    print(colored('Training on 80 frame long clips', 'blue'))

criterion_class = [nn.CrossEntropyLoss().cuda()]  # instead of manual indexing you can use

if config_exp['loss_reg'] == 'L1':
    # nn.CrossEntropyLoss(reduction='mean', ignore_index=-1) and just feed in the score and labels but we don't have
    # anything like this for MSE loss
    criterion_reg = [nn.L1Loss().cuda()]  # [nn.SmoothL1Loss().cuda()]# nn.SmoothL1Loss().cuda() #nn.MSELoss().cuda()
elif config_exp['loss_reg'] == 'SmoothL1':
    criterion_reg = [
        nn.SmoothL1Loss().cuda()]  # [nn.SmoothL1Loss().cuda()]# nn.SmoothL1Loss().cuda() #nn.MSELoss().cuda()
elif config_exp['loss_reg'] == 'MSE':
    criterion_reg = [nn.MSELoss().cuda()]
elif config_exp['loss_reg'] == 'MSEL1':
    criterion_reg = [nn.MSELoss().cuda(), nn.L1Loss().cuda()]
elif config_exp['loss_reg'] == 'MSESmoothL1':
    criterion_reg = [nn.MSELoss().cuda(), nn.SmoothL1Loss().cuda()]

if config_exp['TASK']=='classification':
    config_exp['output_name'] = 'Classification_output.txt'
elif config_exp['TASK'] == 'regression':
    config_exp['output_name'] = 'regression_output.txt'
elif config_exp['TASK'] == 'MTL':
    config_exp['output_name'] = 'MLT_output.txt'
elif config_exp['TASK'] == 'MTL-Emb':
    config_exp['output_name'] = 'MTL-Emb_output.txt'

if not os.path.isdir(config_exp['log_dir']):
    os.makedirs(config_exp['log_dir'])
if not os.path.isdir(config_exp['checkpoint_dir']):
    os.makedirs(config_exp['checkpoint_dir'])
if not os.path.isdir(config_exp['pred_dir']):
    os.makedirs(config_exp['pred_dir'])
if not os.path.isdir(config_exp['CM_dir']):
    os.makedirs(config_exp['CM_dir'])

what_is_different_in_this_code = config_exp['output_name'][:-4]
print(colored(what_is_different_in_this_code, 'magenta'))