import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label=0):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).to(torch.int64)# ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y +1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox

def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0

    l1_reg += l1_reg_all(model.fc_omic)
    l1_reg += l1_reg_all(model.mm)

    return l1_reg


def apply_random_mask(patch_embeddings, percentage):
    _, dim_size, _ = patch_embeddings.shape
    mask_count = int(percentage * dim_size)
    mask = torch.cat([torch.zeros(mask_count), torch.ones(dim_size - mask_count)])
    mask = mask[torch.randperm(dim_size)].unsqueeze(0).unsqueeze(-1)
    return patch_embeddings * mask_count


def init_intra_wsi_loss_function(config):
    if config["intra_modality_mode_wsi"] == "reconstruct_avg_emb" or config[
        "intra_modality_mode_wsi"] == "reconstruct_masked_emb":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = InfoNCE(temperature=config["temperature"])
    return loss_fn


def lable_entropy(pro1, pro2, smooth = 1e-6):

    index1_of_max = torch.argmax(pro1)
    index2_of_max = torch.argmax(pro2)
    class_num =  pro1.shape
    smooth = torch.tensor(smooth).to(pro1.device)
    if index1_of_max == index2_of_max:
        p_mean = (pro1 + pro2) / torch.tensor(2.0).to(pro1.device)
        res = torch.sum(-(p_mean*torch.log(p_mean+smooth) / torch.tensor(np.log(class_num)).to(pro1.device)))

    else:

        p_mean = (pro1 + pro2) / torch.tensor(2.0).to(pro1.device)
        p1 = pro1[index1_of_max]
        p2 = pro2[index2_of_max]
        q = -(p1*torch.log(p1+smooth) + p2*torch.log(p2+smooth)) / torch.tensor(np.log(class_num)).to(pro1.device)

        res = torch.sum(-(p_mean * torch.log(p_mean + smooth) / torch.tensor(np.log(class_num)).to(pro1.device))) + q

    return res


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def InfoNce(query, positive_key, label=0, c=0, que=None, epoch=25, temperature=0.1, reduction='mean', class_num=4,
            dim=256):

    query, positive_key = normalize(query, positive_key)
    positive_logit = query * positive_key

    if c == 1:
        ori_arr = np.array([i for i in range(label, class_num)])
        length = len(ori_arr)
        pro_arr = [1, 0.5, 0.25, 0.125]
        pro_arr = np.array(pro_arr[:length])
        pro_arr = pro_arr / pro_arr.sum()
        label = np.random.choice(ori_arr, size=1, p=pro_arr)

    temp_neg = torch.zeros(class_num - 1, dim).to(query.device)
    temp_neg_pa = torch.zeros(class_num - 1, dim).to(query.device)
    step = 0
    for i in range(class_num):
        if label == i:
            step -= 1
            continue
        else:
            temp_neg[step, :] = que.get_que()[i, :]
            temp_neg_pa[step, :] = que.get_que_pa()[i, :]
        step += 1
    mo = min((1 - 0.5 ** epoch), 0.9)
    que.update_queue(query, label, mo)  # update query=path_embedding,
    que.update_queue_pa(positive_key, label, mo) # update positive_key=omic_embedding

    negative_logits = (temp_neg*query).to(query.device)
    negative_logits_pa = (temp_neg_pa*positive_key).to(query.device)

    # First index in last dimension are the positive samples
    logits = torch.cat([positive_logit, negative_logits], dim=0)
    logits1 = torch.cat([positive_logit, negative_logits_pa], dim=0)
    labels = torch.arange(len(logits), device=query.device)
    labels1 = torch.arange(len(logits1), device=query.device)
    loss = 0.5 * F.cross_entropy(logits / temperature, labels, reduction=reduction) + 0.5 * F.cross_entropy(
        logits1 / temperature, labels1, reduction=reduction)
    return loss


class InfoNCE(nn.Module):

    def __init__(self, temperature=0.1, class_num=4,  dim=256, K=8, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.K = K
        self.class_num = class_num
        self.dim = dim

    def forward(self, query, positive_key,  label = 0, c=0, que=None, temperature=0.1, reduction='mean'):

        query, positive_key = self.normalize(query, positive_key)

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True).T

        # Cosine between all query-negative combinations
        if c==1:
            ori_arr = np.array([i for i in range(label, self.class_num)])
            length = len(ori_arr)
            pro_arr = [1, 0.5, 0.25, 0.125]
            pro_arr = np.array(pro_arr[:length])
            pro_arr = pro_arr / pro_arr.sum()
            label = np.random.choice(ori_arr, size=1, p=pro_arr)

        temp_neg = torch.zeros(self.class_num-1, self.K, self.dim).to(query.device)
        step = 0
        for i in range(self.class_num):
            if label == i:
                step -= 1
                continue
            else:
                temp_neg[step,:,:] = que.get_queue()[i,:,:]
            step += 1
        que.update_queue(positive_key, label)

        negative_logits = torch.einsum('n k d, t d -> n t', temp_neg, query).to(query.device)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=0)
        labels = torch.arange(len(logits), device=query.device)
        loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
        return loss

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


