import torch
import torch.nn as nn
import yaml

try:
    with open('./config_files/config_TUM_exp.yml', 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')
num_class = config['NUMBER_OF_CLASSES']

class RegLoss(nn.Module):
    def __init__(self, model, loss_fn):
        super(RegLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, input, targets):
        reba_pre, _ = self.model(input)
        loss_reg = self.loss_fn[0](reba_pre[targets[1] != -1].view(-1),
                                                 targets[1][targets[1] != -1].view(-1))
        # total_loss = loss_class + loss_reg
        return torch.tensor([0]), loss_reg, loss_reg #loss_reg, total_loss.sum()


class CrossEntLoss(nn.Module):
    def __init__(self, model, loss_fn):
        super(CrossEntLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, input, targets):
        score, _ = self.model(input)
        loss_class = self.loss_fn[0](score[targets[0] != -1].view(-1, num_class),
                                                   targets[0][targets[0] != -1].view(-1))

        return loss_class, torch.tensor(0, device='cuda'), loss_class #loss_reg, total_loss.sum()


class MultiTask2Loss(nn.Module):
    def __init__(self, model, loss_fn):
        super(MultiTask2Loss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eta = nn.Parameter(torch.FloatTensor(len(loss_fn)).fill_(1.), requires_grad=True)  # uniform_(0., 1.)

    def forward(self, input, targets):
        score, reba_pre = self.model(input)
        loss_class = self.eta[0] * self.loss_fn[0](score[targets[0] != -1].view(-1, num_class),
                                                   targets[0][targets[0] != -1].view(-1))
        loss_reg = self.eta[1] * self.loss_fn[1](reba_pre[targets[1] != -1].view(-1),
                                                 targets[1][targets[1] != -1].view(-1))
        total_loss = loss_class + loss_reg
        return loss_class, loss_reg, total_loss.sum()

class MultiTask3Loss(nn.Module):
    def __init__(self, model, loss_fn):
        super(MultiTask3Loss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eta = nn.Parameter(torch.FloatTensor(len(loss_fn)).fill_(1.), requires_grad=True)

    def forward(self, input, targets):
        score, reba_pre = self.model(input)
        loss_class = self.eta[0] * self.loss_fn[0](score[targets[0] != -1].view(-1, num_class),
                                                   targets[0][targets[0] != -1].view(-1))
        loss_reg = self.eta[1] * self.loss_fn[1](reba_pre[targets[1] != -1].view(-1),
                                                 targets[1][targets[1] != -1].view(-1)) + \
                   self.eta[2] * self.loss_fn[2](reba_pre[targets[1] != -1].view(-1),
                                                 targets[1][targets[1] != -1].view(-1))
        total_loss = loss_class + loss_reg
        return loss_class, loss_reg, total_loss.sum()
# class MultiTask3Loss(nn.Module):
#     def __init__(self, model, loss_fn):
#         super(MultiTask3Loss, self).__init__()
#         self.model = model
#         self.loss_fn = loss_fn
#         self.eta = nn.Parameter(torch.FloatTensor(len(loss_fn)).fill_(1.), requires_grad=True)

#     def forward(self, input, targets):
#         score, reba_pre = self.model(input)
#         loss_class = self.eta[0] * self.loss_fn[0](score[targets[0] != -1].view(-1, num_class),
#                                                    targets[0][targets[0] != -1].view(-1))
#         loss_reg = self.eta[1] * self.loss_fn[1](reba_pre[targets[1] != -1].view(-1),
#                                                  targets[1][targets[1] != -1].view(-1)) + \
#                    self.eta[2] * self.loss_fn[2](reba_pre[targets[1] != -1].view(-1),
#                                                  targets[1][targets[1] != -1].view(-1))
#         total_loss = loss_class + loss_reg
#         return loss_class, loss_reg, total_loss.sum()


# class MultiTaskLoss(nn.Module):
#     def __init__(self, tasks):
#         super(MultiTaskLoss, self).__init__()
#         self.tasks = nn.ModuleList(tasks)
#         self.sigma = nn.Parameter(torch.ones(len(tasks)))
#         self.mse = nn.MSELoss()
#
#     def forward(self, x, targets):
#        l = [self.mse(f(x), y) for y, f in zip(targets, self.tasks)]
#        l = 0.5 * torch.Tensor(l) / self.sigma**2
#        l = l.sum() + torch.log(self.sigma.prod())
#        return l