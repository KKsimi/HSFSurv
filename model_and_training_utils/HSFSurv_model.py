from model_and_training_utils.HSFSurv_model_utils import *
import torch.nn as nn
import torch


class Omics_Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, attention_dim)
        self.activation = nn.Tanh()

    def forward(self, features):
        features = self.linear(features)
        attention = self.activation(features)
        attention = torch.softmax(attention, dim=0)
        weighted_features = attention * features
        return weighted_features


class CosAttention(nn.Module):
    def __init__(self):
        super(CosAttention, self).__init__()

    def forward(self, x, y):
        cross_attn = (x * y).softmax(dim=-1) * y
        cross_attn1 = (y * x).softmax(dim=-1) * x
        return cross_attn, cross_attn1


class Split_fea_Net(nn.Module):
    def __init__(self):
        super(Split_fea_Net, self).__init__()
        self.thelt = nn.Sigmoid()

    def forward(self, h_p, h_g):
        h_all = h_p * h_g
        mask = self.thelt(h_all)
        h_inner_p = h_p * mask
        h_split_p = h_p - h_inner_p
        h_inner_g = h_g * mask
        h_split_g = h_g - h_inner_g

        return h_inner_p, h_split_p, h_inner_g, h_split_g


class HSFSurv(nn.Module):
    def __init__(self, device, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600],
                 model_size_wsi: str = 'small',
                 model_size_omic: str = 'small', n_classes=4, dropout=0.25, omic_bag='Attention',
                 proj_ratio=1):
        """
        HSFSurv model Implementation.
        Inspired by https://github.com/mahmoodlab/MCAT/blob/master/models/model_coattn.py
                    https://github.com/Cassie07/PathOmics/blob/main/PathOmics/model_and_training_utils/PathOmics_Survival_model.py

        Args:
            fusion (str): Late fusion method (Choices: concat, bilinear, or None)
            omic_sizes (List): List of sizes of genomic embeddings
            model_size_wsi (str): Size of WSI encoder (Choices: small or large)
            model_size_omic (str): Size of Genomic encoder (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(HSFSurv, self).__init__()
        self.device = device
        self.omic_bag = omic_bag
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256], 'Methylation': [512, 256],
                               'multi_omics': [512, 256]}

        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        self.split_feature = Split_fea_Net()   #MFD Module

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=size[2], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2])])

        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=size[2], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2])])

        ### Path Transformer + Attention Head
        path_encoder_layer1 = nn.TransformerEncoderLayer(d_model=size[2], nhead=8, dim_feedforward=512, dropout=dropout,
                                                         activation='relu')
        self.path_transformer1 = nn.TransformerEncoder(path_encoder_layer1, num_layers=2)
        self.path_attention_head1 = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho1 = nn.Sequential(*[nn.Linear(size[2], size[2])])

        ### Omic Transformer + Attention Head
        omic_encoder_layer1 = nn.TransformerEncoderLayer(d_model=size[2], nhead=8, dim_feedforward=512, dropout=dropout,
                                                         activation='relu')
        self.omic_transformer1 = nn.TransformerEncoder(omic_encoder_layer1, num_layers=2)
        self.omic_attention_head1 = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho1 = nn.Sequential(*[nn.Linear(size[2], size[2])])

        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(
                *[nn.Linear(size[2] * 4, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)

        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

        ### Projection
        self.path_proj = nn.Linear(size[2], int(size[2] * proj_ratio))
        self.omic_proj = nn.Linear(size[2], int(size[2] * proj_ratio))
        self.path_proj1 = nn.Linear(size[2], int(size[2] * proj_ratio))
        self.omic_proj1 = nn.Linear(size[2], int(size[2] * proj_ratio))
        self.omic_cla = nn.Linear(size[2] * 2, n_classes)
        self.path_cla = nn.Linear(size[2] * 2, n_classes)

        self.attention = Attention_Gated(size[2])
        self.dimReduction = DimReduction(size[2]//2, size[2], numLayer_Res=0)

        ### PathOmics - Omics
        self.omics_attention_networks = nn.ModuleList(
            [Omics_Attention(omic_sizes[i], size[2]) for i in range(len(omic_sizes))])

        self.MHA = nn.MultiheadAttention(size[2], 8)
        self.mutiheadattention_networks = nn.ModuleList([self.MHA for i in range(len(omic_sizes))])

    def forward(self, x_path, x_omic, mode='pretrain'):

        # image-bag
        slide_pseudo_feat = []
        numGroup = len(x_omic)

        chunk_list = []
        fea_len = x_path[0].shape[0] // numGroup
        for fe_index in range(numGroup):
            temp = x_path[:, fe_index * fea_len:(fe_index + 1) * fea_len]
            chunk_list.append(temp)

        for fea_li in chunk_list:
            subFeat_tensor = fea_li
            tmidFeat = self.dimReduction(subFeat_tensor)  # n x 256
            tAA = self.attention(tmidFeat).squeeze(0)  # n  ##
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x 256
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x 256
            af_inst_feat = tattFeat_tensor  # 1 x 256
            slide_pseudo_feat.append(af_inst_feat)

        h_path_bag = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x 256

        if self.omic_bag == 'Attention':
            h_omic = [self.omics_attention_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        elif self.omic_bag == 'SNN_Attention':
            h_omic = []
            for idx, sig_feat in enumerate(x_omic):
                snn_feat = self.sig_networks[idx].forward(sig_feat)
                snn_feat = snn_feat.unsqueeze(0).unsqueeze(1)
                attention_feat, _ = self.mutiheadattention_networks[idx].forward(snn_feat, snn_feat, snn_feat)
                h_omic.append(attention_feat.squeeze())
        else:
            h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in
                      enumerate(x_omic)]  ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic)  ### numGroup x 256

        h_inner_p, h_split_p, h_inner_g, h_split_g = self.split_feature(h_path_bag, h_omic_bag)

        h_path_trans = self.path_transformer(h_inner_p)
        h_omic_trans = self.omic_transformer(h_inner_g)

        h_path_trans1 = self.path_transformer1(h_split_p)
        h_omic_trans1 = self.omic_transformer1(h_split_g)

        if mode == 'pretrain':
            A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
            A_path = torch.transpose(A_path, 1, 0)
            h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
            h_path_fea = self.path_proj(h_path).squeeze()

            A_path1, h_path1 = self.path_attention_head1(h_path_trans1.squeeze(1))
            A_path1 = torch.transpose(A_path1, 1, 0)
            h_path1 = torch.mm(F.softmax(A_path1, dim=1), h_path1)
            h_path_fea1 = self.path_proj1(h_path1).squeeze()

            h_path_all = torch.cat([h_path_fea, h_path_fea1], dim=0)
            h_path_cla = self.path_cla(h_path_all).squeeze()
            h_path_cla = F.softmax(h_path_cla, dim=0)

        else:
            A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
            A_path = torch.transpose(A_path, 1, 0)
            h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
            h_path_fea = self.path_rho(h_path).squeeze()

            A_path1, h_path1 = self.path_attention_head1(h_path_trans1.squeeze(1))
            A_path1 = torch.transpose(A_path1, 1, 0)
            h_path1 = torch.mm(F.softmax(A_path1, dim=1), h_path1)
            h_path_fea1 = self.path_rho1(h_path1).squeeze()

        if mode == 'pretrain':
            A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
            A_omic = torch.transpose(A_omic, 1, 0)
            h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
            h_omic_fea = self.omic_proj(h_omic).squeeze()

            A_omic1, h_omic1 = self.omic_attention_head1(h_omic_trans1.squeeze(1))
            A_omic1 = torch.transpose(A_omic1, 1, 0)
            h_omic1 = torch.mm(F.softmax(A_omic1, dim=1), h_omic1)
            h_omic_fea1 = self.omic_proj1(h_omic1).squeeze()

            h_omic_all = torch.cat([h_omic_fea, h_omic_fea1], dim=0)
            h_omic_cla = self.omic_cla(h_omic_all).squeeze()
            h_omic_cla = F.softmax(h_omic_cla, dim=0)

        else:
            A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
            A_omic = torch.transpose(A_omic, 1, 0)
            h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
            h_omic_fea = self.omic_rho(h_omic).squeeze()

            A_omic1, h_omic1 = self.omic_attention_head1(h_omic_trans1.squeeze(1))
            A_omic1 = torch.transpose(A_omic1, 1, 0)
            h_omic1 = torch.mm(F.softmax(A_omic1, dim=1), h_omic1)
            h_omic_fea1 = self.omic_rho1(h_omic1).squeeze()

        if mode == 'pretrain':
            if len(h_path_fea.shape) == 1:
                h_path_fea = h_path_fea.unsqueeze(0)
            if len(h_omic_fea.shape) == 1:
                h_omic_fea = h_omic_fea.unsqueeze(0)
            return h_path_fea, h_path_cla, h_omic_fea, h_omic_cla

        else:
            if self.fusion == 'bilinear':
                h = self.mm(h_path_fea.unsqueeze(dim=0), h_omic_fea.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path_fea, h_path_fea1, h_omic_fea, h_omic_fea1], axis=0))
            elif self.fusion == 'image':
                h = h_path_fea.squeeze()
            elif self.fusion == 'omics':
                h = h_omic_fea.squeeze()

            ### Survival Layer
            logits = self.classifier(h).unsqueeze(0)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)


            return hazards, S, Y_hat, None, logits


# from MCAT_models.model_utils
def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
