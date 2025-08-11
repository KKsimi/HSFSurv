from torch.utils.data import DataLoader
from data_loading_utils.Customized_Dataset import *
from model_and_training_utils.HSFSurv_model import *
from model_and_training_utils.train_and_eval_utils_core import *
from model_and_training_utils.Customized_Loss import *
from model_and_training_utils.help_utils import *
import torch.optim as optim
import torch
from sklearn.model_selection import KFold, train_test_split
import time
import copy
import random
import argparse
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')

parser.add_argument('--omic_modal', choices=['mRNA', 'Methylation'], default='mRNA', type=str,
                    help='omic model')
parser.add_argument('--kfold_split_seed', default=42, type=int, help='random seed')
parser.add_argument('--feature_dimension', choices=[256, 512, 1024, 2048], default=1024, type=int,
                    help='image patch feature dimension')
parser.add_argument('--n_bins', default=4, type=int, help='n survival event intervals')
parser.add_argument('--eps', default=1e-6, type=float, help='calculate bins')
parser.add_argument('--finetune_epochs', default=40, type=int, help='finetuning epochs')
parser.add_argument('--pretrain_epochs', default=20, type=int, help='pretrain epochs')
parser.add_argument('--pretrain_lr', default=2e-4, type=float, help='pretraining learning rate')
parser.add_argument('--finetune_lr', default=2e-4, type=float, help='finetuningg learning rate')
parser.add_argument('--wd', default=0, type=float, help='weight decay')
parser.add_argument('--wd_ft', default=0, type=float, help='weight decay')
parser.add_argument('--gradient_accumulation_steps', default=32, type=int, help='Gradient Accumulation Step for MCAT')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--bag_loss', choices=['ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', type=str,
                    help='supervised loss')
parser.add_argument('--alpha_surv', default=0.0, type=float, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', default='None',
                    help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--model_type', default='HSFSurv', help='model name')
parser.add_argument('--model_fusion_type', choices=['concat', 'bilinear'], default='concat',
                    help='model fusion type for finetuning (no pretraining and finetuning in baseline models on same dataset setting (e.g., TCGA-COAD experiments))')
parser.add_argument('--model_pretrain_fusion_type', choices=['concat', 'bilinear'], default='concat',
                    help='model fusion type for pretraining (no pretraining and finetuning in baseline models on same dataset setting (e.g., TCGA-COAD experiments))')
parser.add_argument('--less_data_ratio', default=0.1, type=float, help='less data ratio')
parser.add_argument('--results_string',
                    help='pretraining results (using for loading pretraining model). Example: c-index value_variance among folds')  #
parser.add_argument('--cuda_device', default='0', type=str)
parser.add_argument('--gene_family_info_path', default='gene_family', help='gene family info')
parser.add_argument('--dataset_name', choices=['LUAD', 'KIRC', 'BLCA', 'STAD', 'COAD'], default='LUAD')
parser.add_argument('--feature_folder', default='/remote-home/hhhhh/Path/',  help='extracted feature folder path')
parser.add_argument('--clinical_path', default='/remote-home/hhhhh/Path/',  help='clinical info folder path')
parser.add_argument('--proj_ratio', default=1, type=int,
                    help='Model architecture related param: 1: (256, 256), 0.5: (256, 128)')
parser.add_argument('--omic_bag_type', choices=['Attention', 'SNN', 'SNN_Attention'], default='SNN',
                    help='Architecture for omics data feature extraction')
parser.add_argument('--save_model_folder_name', default='model_checkpoint', help='Folder name')
parser.add_argument('--experiment_folder_name', default='main_experiments', help='Experiment name')
parser.add_argument('--experiment_id', default='1', type=str)
parser.add_argument('--pretrain_loss', choices=['MSE', 'Cosine', 'InfoNCE'], default='InfoNCE', type=str)
parser.add_argument('--load_model_finetune', default=False, type=bool,
                    help='whether in finetuning mode, if so, no need for running pretraining code')
parser.add_argument('--less_data', action='store_true',
                    help='whether use fewer data for model finetuning(no pretraining and finetuning in baseline models on same dataset setting (e.g., TCGA-COAD experiments))')  # default False (not add)
parser.add_argument('--seperate_test', action='store_true', help='Use one fold as test set, do not use it for training or validation')
parser.add_argument('--parameter_reset', action='store_true',
                    help='During finetuning, whether reset pretrained parameters or not')


def main():
    params = parser.parse_args()
    print()
    print(params.load_model_finetune)
    print(params.parameter_reset)
    dataset_name = params.dataset_name
    params.feature_folder += dataset_name + '-feature/features/pt_files/'
    params.clinical_path += dataset_name + '-gene/' + dataset_name.lower() + '_tcga_pan_can_atlas_2018/data_clinical_patient.csv'
    # The data_clinical_patien.csv file based on data_clinical_patien.txt. Please refer to lines 170 to 186 of the code for the specific content.
    # Insert the label into the CSV file according to the --n_bins using lines 170–186 of the provided code.
    print(params.feature_folder)
    print(params.clinical_path)

    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.manual_seed(params.kfold_split_seed)
    torch.cuda.manual_seed(params.kfold_split_seed)
    np.random.seed(params.kfold_split_seed)
    random.seed(params.kfold_split_seed)

    start = time.time()

    # Changable parameters
    if params.omic_modal == 'Methylation':
        model_size_omic = 'Methylation'
    else:
        model_size_omic = 'small'

    if params.omic_modal == 'mRNA':
        z_score_path = '/remote-home/hhhhh/Path/' + dataset_name + '-gene/' + dataset_name.lower() + '_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt'

    elif params.omic_modal == 'Methylation':
        z_score_path = '/remote-home/hhhhh/Path/' + dataset_name + '-gene/' + dataset_name.lower() + '_tcga_pan_can_atlas_2018/data_methylation_hm27_hm450_merged.txt'

    load_path = '/remote-home/hhhhh/code_map/HSFSurv/{}/{}/{}_model_{}_OmicBag_{}_FusionType_{}_OmicType_{}'.format(
        params.save_model_folder_name, params.experiment_folder_name, params.experiment_id, params.model_type,
        params.omic_bag_type, params.model_pretrain_fusion_type, params.omic_modal)

    if params.load_model_finetune:
        save_path = load_path
    else:
        save_path = '/remote-home/hhhhh/code_map/HSFSurv/{}/{}/'.format(params.save_model_folder_name,
                                                                      params.experiment_folder_name)
        create_directory(save_path)
        save_path += '/{}_model_{}_OmicBag_{}_FusionType_{}_OmicType_{}'.format(params.experiment_id, params.model_type,
                                                                                params.omic_bag_type,
                                                                                params.model_fusion_type,
                                                                                params.omic_modal)
        create_directory(save_path)

    if params.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=params.alpha_surv)
    elif params.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=params.alpha_surv)
    elif params.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError

    if params.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif params.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print()
    if params.seperate_test:
        k_fold = 5
        fold_mode = 'train_val_test'
    else:
        k_fold = 5
        fold_mode = 'k_fold'
    print()

    patient_id_list = get_overlapped_patient(params.feature_folder, params.clinical_path, z_score_path)
    patient_id_list = sorted(patient_id_list)


    # load genomics z-score
    if params.omic_modal == 'mRNA':
        df_gene = pd.read_csv(z_score_path, delimiter="\t")
        df_gene = df_gene.drop(['Entrez_Gene_Id'], axis=1)
        df_gene = df_gene[df_gene['Hugo_Symbol'].notna()].dropna()
        df_gene = df_gene.set_index('Hugo_Symbol')
    else:
        # load methylation
        df_methylation = pd.read_csv(z_score_path, delimiter="\t")
        df_methylation = df_methylation.drop(['ENTITY_STABLE_ID'], axis=1)
        df_methylation = df_methylation.drop(['DESCRIPTION'], axis=1)
        df_methylation = df_methylation.drop(['TRANSCRIPT_ID'], axis=1)
        df_methylation = df_methylation[df_methylation['NAME'].notna()].dropna()
        df_methylation = df_methylation.set_index('NAME')
        df_gene = df_methylation

    # load gene family and genes info
    dict_family_genes = load_gene_family_info('gene_family')

    # load clinical info
    df_clinical = pd.read_csv(params.clinical_path)

    '''
    disc_labels: https://github.com/mahmoodlab/MCAT/blob/b9cca63be83c67de7f95308d54a58f80b78b0da1/datasets/dataset_survival.py
    '''

    df_uncensored = df_clinical[df_clinical.OS_STATUS == '1:DECEASED']
    disc_labels, q_bins = pd.qcut(df_uncensored['OS_MONTHS'], q=params.n_bins, retbins=True, labels=False)
    q_bins[-1] = df_clinical['OS_MONTHS'].max() + params.eps
    q_bins[0] = df_clinical['OS_MONTHS'].min() - params.eps
    df_clinical = df_clinical.drop(df_clinical[df_clinical['label'] < 0].index)

    '''The comment content is to generate the data_clinical_patien.csv file based on data_clinical_patien.txt, then replace
     params.clinical_path with the path where the csv was generated, and finally comment out the following four lines of code.'''

    # disc_labels, q_bins = pd.cut(df_clinical['OS_MONTHS'], bins = q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    # df_clinical.insert(2, 'label', disc_labels.values.astype(int))
    # df_clinical = df_clinical.drop(df_clinical[df_clinical['OS_MONTHS']==0].index)
    # df_clinical = df_clinical.dropna(subset=['OS_MONTHS'])


    best_c_index_list = []

    kf = KFold(n_splits=k_fold, random_state=params.kfold_split_seed, shuffle=True)

    finetune_result_by_fold = {i: [] for i in range(k_fold)}


    if params.seperate_test:
        patient_id_list, test_patient_id = train_test_split(patient_id_list, test_size=0.2,
                                                            random_state=params.kfold_split_seed)
        seperate_test_dataset = CustomizedDataset(test_patient_id, df_gene, df_clinical, dict_family_genes,
                                                  params.feature_dimension, params.feature_folder)
        seperate_test_loader = DataLoader(seperate_test_dataset, batch_size=params.batch_size, shuffle=False,
                                          num_workers=12)

    for fold, (train_idx, test_idx) in enumerate(kf.split(patient_id_list)):
        print()
        print('【 Fold {} 】'.format(fold + 1))

        if params.less_data:
            finetune_train_idx = copy.deepcopy(train_idx)
            random.seed(params.kfold_split_seed)
            finetune_train_idx = list(finetune_train_idx)
            num_data = int(len(finetune_train_idx) * params.less_data_ratio)
            finetune_train_idx = random.sample(finetune_train_idx, k=num_data)
            finetune_train_idx = np.array(finetune_train_idx)
            finetune_train_patient_id = list(np.array(patient_id_list)[finetune_train_idx])
            finetune_train_dataset = CustomizedDataset(finetune_train_patient_id, df_gene, df_clinical,
                                                       dict_family_genes, params.feature_dimension,
                                                       params.feature_folder)
            finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=params.batch_size, shuffle=True,
                                               num_workers=12)

        train_patient_id = list(np.array(patient_id_list)[train_idx])
        test_patient_id = list(np.array(patient_id_list)[test_idx])
        print('Number of patient for model training : {}'.format(len(train_patient_id)))
        print('Number of patient for model testing : {}'.format(len(test_patient_id)))


        train_dataset = CustomizedDataset(train_patient_id, df_gene, df_clinical, dict_family_genes,
                                          params.feature_dimension, params.feature_folder)
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                  num_workers=12)

        test_dataset = CustomizedDataset(test_patient_id, df_gene, df_clinical, dict_family_genes,
                                         params.feature_dimension, params.feature_folder)
        test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=12)

        x_path, x_omics, censorship, survival_months, label, patient_id = next(iter(test_loader))


        '''
        Model and hyper-parameter
        '''

        # Model
        device = "cuda:" + str(params.cuda_device) if torch.cuda.is_available() else "cpu"
        if params.model_type == 'HSFSurv':
            model = HSFSurv(device=device, fusion=params.model_fusion_type, n_classes=params.n_bins,
                                   omic_sizes=[i.shape[1] for i in x_omics], model_size_omic=model_size_omic,
                                   omic_bag=params.omic_bag_type, proj_ratio=params.proj_ratio)

        if not params.load_model_finetune:
            print()
            print('[Train from scratch]')
            model.to(device)
            print(device)

            '''
            pretrain

            '''

            pretrain_optimizer = optim.Adam(model.parameters(), lr=params.pretrain_lr, weight_decay=params.wd)

            if params.pretrain_loss == 'MSE':
                loss_fn_pretrain = torch.nn.MSELoss()
            elif params.pretrain_loss == 'Cosine':
                loss_fn_pretrain = torch.nn.CosineEmbeddingLoss()
            elif params.pretrain_loss == 'InfoNCE':
                loss_fn_pretrain = 'Info'
            logs, best_pretrain_model = train_and_evaluate(fold, params.pretrain_epochs, model,
                                                           [train_loader, test_loader, seperate_test_loader],
                                                           pretrain_optimizer,
                                                           loss_fn_pretrain, reg_fn, device, save_path, lambda_reg=0.,
                                                           gc=params.gradient_accumulation_steps, save_model=False,
                                                           seperate_test_mode=params.seperate_test,
                                                           model_mode='pretrain', fold_mode=fold_mode)


        else:
            print()
            print('[Load pretrained model]')
            model_dict = model.state_dict()
            checkpoint = torch.load(load_path + '/fold_{}_pretrain_model.pt'.format(fold))
            pretrained_dict = checkpoint['model_state_dict']

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(pretrained_dict)

            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     best_pretrain_model = torch.nn.DataParallel(model)
            best_pretrain_model = model
            best_pretrain_model.to(device)

        '''
        Finetune

        '''
        print()
        print('Finetune')

        if params.less_data:
            train_loader = finetune_train_loader
            print('Number of patient for model finetune-training : {}'.format(len(finetune_train_patient_id)))
            print('Number of patient for model finetune-testing : {}'.format(len(test_patient_id)))
        else:
            print('Number of patient for model finetune-training : {}'.format(len(train_patient_id)))
            print('Number of patient for model finetune-testing : {}'.format(len(test_patient_id)))
        print()


        finetune_optimizer = optim.Adam(best_pretrain_model.parameters(), lr=params.finetune_lr,
                                        weight_decay=params.wd_ft)

        if params.seperate_test:
            logs, best_test_c_index, best_epoch = train_and_evaluate(fold, params.finetune_epochs, best_pretrain_model,
                                                                     [train_loader, test_loader, seperate_test_loader],
                                                                     finetune_optimizer, loss_fn, reg_fn, device,
                                                                     save_path, lambda_reg=0.,
                                                                     gc=params.gradient_accumulation_steps,
                                                                     save_model=True,
                                                                     seperate_test_mode=params.seperate_test,
                                                                     model_mode='Finetune', fold_mode=fold_mode)
        else:
            logs, best_test_c_index, best_epoch = train_and_evaluate(fold, params.finetune_epochs, best_pretrain_model,
                                                                     [train_loader, test_loader], finetune_optimizer,
                                                                     loss_fn, reg_fn, device, save_path, lambda_reg=0.,
                                                                     gc=params.gradient_accumulation_steps,
                                                                     save_model=True,
                                                                     seperate_test_mode=params.seperate_test,
                                                                     model_mode='Finetune', fold_mode=fold_mode)

        print('【 in fold {}, {} epoch 】 The highest test c-index: {}'.format(fold + 1, best_epoch, best_test_c_index))
        best_c_index_list.append(best_test_c_index)
        finetune_result_by_fold[fold].append(best_epoch)
        finetune_result_by_fold[fold].append(best_test_c_index)

        save = pd.DataFrame.from_dict(logs)
        if not params.load_model_finetune:
            save.to_csv(save_path + '/fold{}_Train_test_logs.csv'.format(fold + 1), header=True)
        else:
            if params.less_data:
                save.to_csv(save_path + '/{}_fold{}_Train_test_logs.csv'.format(params.less_data_ratio, fold + 1),
                            header=True)

            if params.model_fusion_type != 'concat':
                save.to_csv(save_path + '/{}_fold{}_Train_test_logs.csv'.format(params.model_fusion_type, fold + 1),
                            header=True)

    print()
    print_result_as_table(finetune_result_by_fold, k_fold)
    print('【 The cross-validated c-index (average c-index across folds) 】: {}'.format(
        sum(best_c_index_list) / len(best_c_index_list)))
    print('【Experiment id】: {}'.format(params.experiment_id))
    print()
    print('【 The standard deviation among folds 】: {}'.format(np.std(best_c_index_list)))
    print()
    finetune_result_by_fold['avg'] = sum(best_c_index_list) / len(best_c_index_list)
    finetune_result_by_fold['std'] = np.std(best_c_index_list)
    res = pd.DataFrame.from_dict(finetune_result_by_fold)
    res.to_csv(save_path + '/result.csv', header=True)
    print('finish')
    stop = time.time()
    hr = (stop - start) / 3600
    print('training time : {} hrs'.format(hr))


if __name__ == "__main__":
    main()



