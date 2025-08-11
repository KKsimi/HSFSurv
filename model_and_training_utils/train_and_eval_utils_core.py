import random
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os 
import torch
from sksurv.metrics import concordance_index_censored
from model_and_training_utils.Customized_Loss import lable_entropy, InfoNce


def create_directory(save_path):
    isExist = os.path.exists(save_path)
    if not isExist:
        os.mkdir(save_path)
        
def train_and_evaluate(fold, epochs, model, loader_list, optimizer, loss_fn, reg_fn, device, save_path, lambda_reg=0., gc = 5, save_model = False, seperate_test_mode = False, model_mode = 'pretrain', fold_mode = 'train_val_test'):
    """
    Based on: https://github.com/Cassie07/PathOmics/blob/main/PathOmics/model_and_training_utils/train_and_eval_utils_core.py
    """
    if fold_mode == 'train_val_test':
        if model_mode != 'pretrain':
            logs = {'train_loss':[], 'train_c_index':[], 'test_loss':[], 'test_c_index':[]}
        else:
            logs = {'train_loss':[],'test_loss':[]}
    else:
        if model_mode != 'pretrain':
            logs = {'train_loss':[], 'train_c_index':[], 'test_loss':[], 'test_c_index':[]}
        else:
            logs = {'train_loss':[], 'test_loss':[]}
            
    best_val_loss = float('inf')
    best_val_c_index = float('-inf')
    best_test_c_index = float('-inf')
    best_epoch = 0
    prev_test_loss = float('inf')
    
    if fold_mode == 'train_val_test':
        train_loader, val_loader, test_loader = loader_list
    elif fold_mode == 'k_fold' and model_mode != 'pretrain' and seperate_test_mode:
        train_loader, val_loader, test_loader = loader_list
    else:
        train_loader, test_loader = loader_list

    que1 = Queue_con()

    for epoch in range(epochs):
        print()
        if model_mode == 'pretrain':
            train_loss = train_loop(model, train_loader, optimizer, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, gc = gc, model_mode = model_mode, que=que1)
            print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))
            logs['train_loss'].append(train_loss)
            test_loss = test_loop(model, val_loader, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, model_mode = model_mode, que=que1)
            if abs(test_loss) < abs(prev_test_loss):
                print('Epoch: {}, test_loss: {:.4f}'.format(epoch, test_loss))
                prev_test_loss = test_loss
                best_epoch = epoch
                best_pretrain_model = model
                print("Save a new model")
                if save_model:
                    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, save_path + '/' + 'fold_{}_model.pt'.format(fold))
                logs['test_loss'].append(test_loss)
            else:
                logs['test_loss'].append(0)
       
        else:
            train_loss_surv, train_loss, train_c_index = train_loop(model, train_loader, optimizer, loss_fn, reg_fn, device, epoch, lambda_reg=0., gc = gc, model_mode = model_mode)
            print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, train_c_index))

            logs['train_loss'].append(train_loss)
            logs['train_c_index'].append(train_c_index)
            
            if not seperate_test_mode:
                test_loss_surv, test_loss, test_c_index = test_loop(model, test_loader, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, model_mode = model_mode)
                if test_c_index > best_test_c_index:
                    print('Epoch: {}, test_loss_surv: {:.4f}, test_c_index: {:.4f}'.format(epoch, test_loss, test_c_index))
                    best_test_c_index = test_c_index
                    best_epoch = epoch
                    print("Save a new model")
                    if save_model:
                        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, save_path + '/' + 'fold_{}_finetune_model.pt'.format(fold))
                    logs['test_loss'].append(test_loss)
                    logs['test_c_index'].append(test_c_index)
                else:
                    logs['test_loss'].append(0)
                    logs['test_c_index'].append(0)
            else:
                val_loss_surv, val_loss, val_c_index = test_loop(model, val_loader, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg, model_mode = model_mode)
                if val_c_index > best_val_c_index:
                    print('Epoch: {}, val_loss_surv: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, val_c_index))
                    test_loss_surv, test_loss, test_c_index = test_loop(model, test_loader, loss_fn, reg_fn, device, epoch, lambda_reg=lambda_reg,  model_mode = model_mode)

                    if test_c_index > best_test_c_index:
                        print('Epoch: {}, test_loss_surv: {:.4f}, test_c_index: {:.4f}'.format(epoch, test_loss, test_c_index))
                        best_test_c_index = test_c_index
                        best_epoch = epoch
                        print("Save a new model")
                        if save_model:
                            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, save_path + '/' + 'fold_{}_finetune_model.pt'.format(fold))
                        logs['test_loss'].append(test_loss)
                        logs['test_c_index'].append(test_c_index)
                    else:
                        logs['test_loss'].append(0)
                        logs['test_c_index'].append(0)
                        
                else:
                    logs['test_loss'].append(0)
                    logs['test_c_index'].append(0)

    if model_mode == 'pretrain':
        return logs, best_pretrain_model
    else:
        return logs, best_test_c_index, best_epoch

            
def train_loop(model, loader, optimizer, loss_fn, reg_fn, device, epoch, que=None, lambda_reg=1e-4, gc = 16, model_mode = 'pretrain'):


    model.train()
    train_loss_surv, train_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx,(data_WSI, data_omic, c, event_time, label, patient_id) in enumerate(tqdm(loader)):
        
        data_WSI = data_WSI.squeeze().to(device)
        data_omic = [i.squeeze().to(device) for i in data_omic]
        lab = label.int()
        label = label.to(device)
        cen = c
        c = c.to(device)
        target = torch.tensor([1])
        target = target.to(device)


        if model_mode == 'pretrain':
            path_embedding, path_cla, omic_embedding, omic_cla = model(x_path=data_WSI, x_omic=data_omic, mode = model_mode)

            loss1 = InfoNce(query=path_embedding, positive_key=omic_embedding, label=lab, c=cen, que=que, epoch=epoch)
            loss2 = lable_entropy(path_cla, omic_cla, smooth = 1e-6)
            loss = 0.5*loss1 + loss2
            train_loss += loss.item()

        else:
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic, mode = model_mode)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg
                loss_reg = loss_reg.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            train_loss_surv += loss_value
            train_loss += loss_value + loss_reg

            loss = loss / gc + loss_reg
            
        # backward pass    
        loss.backward(retain_graph=True)

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()
            
    if model_mode == 'pretrain': 
        return train_loss/len(loader)
    else:
        # calculate loss and error for epoch
        train_loss_surv /= len(loader)
        train_loss /= len(loader)

        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        return train_loss_surv, train_loss, c_index

def test_loop(model, loader, loss_fn, reg_fn, device, epoch, que=None, lambda_reg=0., model_mode = 'pretrain'):

    
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx,(data_WSI, data_omic, c, event_time, label, patient_id) in enumerate(tqdm(loader)):

        data_WSI = data_WSI.squeeze().to(device)
        data_omic = [i.squeeze().to(device) for i in data_omic]
        lab = label.int()
        label = label.to(device)
        cen = c
        c = c.to(device)
        target = torch.tensor([1])
        target = target.to(device)

        with torch.no_grad():
            
            if model_mode == 'pretrain':
                path_embedding, path_cla, omic_embedding, omic_cla = model(x_path=data_WSI, x_omic=data_omic,  mode = model_mode)

            else:
                hazards, S, Y_hat, _, logits = model(x_path=data_WSI, x_omic=data_omic,  mode = model_mode)

        if model_mode == 'pretrain':

            loss1 = InfoNce(query=path_embedding, positive_key=omic_embedding, label=lab, c=cen, que=que, epoch=epoch)
            loss2 = lable_entropy(path_cla, omic_cla, smooth=1e-6)
            loss = 0.5*loss1 + loss2
            val_loss += loss.item()

        else:
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg
                loss_reg = loss_reg.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            val_loss_surv += loss_value
            val_loss += loss_value + loss_reg

    if model_mode == 'pretrain':
        return val_loss/len(loader)

    else:
        # calculate loss and error for epoch
        val_loss_surv /= len(loader)
        val_loss /= len(loader)

        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        return val_loss_surv, val_loss, c_index


class Queue_con(nn.Module):
    def __init__(self, class_num=4, dim=256):
        super().__init__()
        self.class_num = class_num
        self.dim = dim
        self.que = torch.randn(class_num,  dim)
        self.que_pa = torch.randn(class_num,  dim)

    @torch.no_grad()
    def update_queue(self, keys, index, m=0.9):
        temp = self.que[index, :] * torch.tensor(m, dtype=float)
        self.que[index, :] = torch.tensor(1-m, dtype=float) * keys.unsqueeze(0).cpu() + temp
        return self.que

    @torch.no_grad()
    def update_queue_pa(self, keys, index, m=0.9):
        temp = self.que_pa[index, :] * torch.tensor(m, dtype=float)
        self.que_pa[index, :] = torch.tensor(1-m, dtype=float) * keys.unsqueeze(0).cpu() + temp
        return self.que_pa

    @torch.no_grad()
    def get_que(self):
        return self.que

    @torch.no_grad()
    def get_que_pa(self):
        return self.que_pa
