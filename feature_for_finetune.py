from src.engine.evaluator import get_and_print_results
import numpy as np
import os

auroc_list, aupr_list, fpr_list = [], [], []

# datasets = ['cotton', 'mango', 'strawberry', 'pvtc', 'plant_village']
# splits = ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_all']
# methods = ['energy', 'entropy', 'var', 'msp', 'max_logits']

# datasets = ['pvt', 'herbarium_19']
datasets = ['plant_village',]
splits = ['_1', '_2', '_3', '_4', '']
methods = ['energy', 'entropy', 'var', 'msp', 'max_logits']

def cosine_similarity_matrix(matrix1, matrix2):
    # similarity_matrix = np.dot(norm(matrix1), norm(matrix2).T)    # 归一化的测试样本与训练样本之间的余弦相似度
    similarity_matrix = np.dot(matrix1, matrix2.T)
    return similarity_matrix

def norm(matrix1):
    return matrix1 / np.linalg.norm(matrix1, axis=1, keepdims=True)     # 归一化

def norm_score(ind_scores, ood_scores):
    '根据测试集的logits进行归一化'
    min_value = min(ind_scores) # np.minimum(id_scores.min(), ood_scores.min())
    max_value = max(ind_scores) # np.maximum(id_scores.max(), ood_scores.max())
    gap = max_value - min_value
    # 将数据归一化到[0,1]
    ind_scores_norm = (ind_scores - min_value) / gap
    ood_scores_norm = (ood_scores - min_value) / gap
    # ood_scores_norm = np.where(ood_scores_norm < 0, 0, np.where(ood_scores_norm > 2, 2, ood_scores_norm))

    return ind_scores_norm, ood_scores_norm


def kNN_OOD(root, name):
    ''' implementation of KNN '''
    InD_train_dataset_start_path = f'{root}run2/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_start_path  = f'{root}run2/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_start_path       = f'{root}run2/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径
    # 使用np.load()函数读取npy文件
    InD_train_start_dataset = np.load(InD_train_dataset_start_path)
    InD_test_start_dataset = np.load(InD_test_dataset_start_path)
    OOD_start_dataset = np.load(OOD_dataset_start_path)

    InD_similarity = cosine_similarity_matrix(InD_test_start_dataset, InD_train_start_dataset)
    OOD_similarity = cosine_similarity_matrix(OOD_start_dataset, InD_train_start_dataset)

    InD_concat_logits = 1 - np.max(InD_similarity, axis=1)
    OOD_concat_logits = 1 - np.max(OOD_similarity, axis=1)

    return InD_concat_logits, OOD_concat_logits


# for dataset in datasets:
#     if dataset == 'cifar100': names = ['tiny-imagenet',]
#     else: names = ['SUN', 'iNaturalist', 'Places', 'dtd']
#     for split in splits:
#         for pre in ['sup_vitb16_imagenet21k']:#'mae_vitb16',
#             for name in names:
#                 root = f'/data/Jiuqing/Enhance_OOD_plant_new_split/Finetune_OOD/{dataset}{split}/{pre}/lr0.0_wd0.0/'
#                 print(root, name)
#                 InD_concat_logits, OOD_concat_logits = kNN_OOD(root, name)
#                 get_and_print_results(None, InD_concat_logits, OOD_concat_logits, auroc_list, aupr_list, fpr_list)



def Diff_kNN_OOD(root, root0, name):
    ''' two KNN ensamble'''   # 两个knn集成
    InD_train_dataset_start_path = f'{root0}run1/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_start_path = f'{root0}run1/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_start_path = f'{root0}run1/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径

    InD_train_dataset_end_path = f'{root}run1/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_end_path = f'{root}run1/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_end_path = f'{root}run1/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径

    # 使用np.load()函数读取npy文件
    InD_train_start_dataset = np.load(InD_train_dataset_start_path)
    InD_test_start_dataset = np.load(InD_test_dataset_start_path)
    OOD_start_dataset = np.load(OOD_dataset_start_path)

    InD_train_end_dataset = np.load(InD_train_dataset_end_path)
    InD_test_end_dataset = np.load(InD_test_dataset_end_path)
    OOD_end_dataset = np.load(OOD_dataset_end_path)

    # # # 归一化
    InD_train_start_dataset = norm(InD_train_start_dataset)
    InD_test_start_dataset  = norm(InD_test_start_dataset)
    OOD_start_dataset       = norm(OOD_start_dataset)
    InD_train_end_dataset   = norm(InD_train_end_dataset)
    InD_test_end_dataset    = norm(InD_test_end_dataset)
    OOD_end_dataset         = norm(OOD_end_dataset)

    InD_similarity_start = cosine_similarity_matrix(InD_test_start_dataset, InD_train_start_dataset)
    OOD_similarity_start = cosine_similarity_matrix(OOD_start_dataset, InD_train_start_dataset)

    InD_similarity_end = cosine_similarity_matrix(InD_test_end_dataset, InD_train_end_dataset)
    OOD_similarity_end = cosine_similarity_matrix(OOD_end_dataset, InD_train_end_dataset)

    InD_similarity_sum = InD_similarity_end + InD_similarity_start
    OOD_similarity_sum = OOD_similarity_end + OOD_similarity_start
    InD_logits_sum = 2 - np.max(InD_similarity_sum, axis=1)
    OOD_logits_sum = 2 - np.max(OOD_similarity_sum, axis=1)

    InD_similarity_end = 1 - np.max(InD_similarity_end, axis=1)
    OOD_similarity_end = 1 - np.max(OOD_similarity_end, axis=1)

    return InD_logits_sum, OOD_logits_sum , InD_similarity_end, OOD_similarity_end


# for dataset in datasets:
#     names = ['tiny-imagenet',] if dataset == 'cifar100' else ['SUN', 'iNaturalist', 'Places', 'dtd']
#     for split in splits:
#         for pre in ['sup_vitb16_imagenet21k']:#'mae_vitb16',
#             for name in names:
#                 root = f'/data/Jiuqing/Enhance_OOD_plant_new_split/Finetune_OOD/{dataset}{split}/{pre}/lr0.0_wd0.0/'
#                 print(root, name)
#                 InD_logits_sum, OOD_logits_sum, InD_logits_diff, OOD_logits_diff = Diff_kNN_OOD(root, name)
#                 # print("diff")
#                 get_and_print_results(None, InD_logits_diff, OOD_logits_diff, auroc_list, aupr_list, fpr_list)
#                 # print("sum")
#                 get_and_print_results(None,  InD_logits_sum, OOD_logits_sum, auroc_list, aupr_list, fpr_list)



def concat_kNN_OOD(root, root0, name):
    ''' concate the feature and using KNN '''   # 特征合并后，使用kNN方法
    InD_train_dataset_start_path = f'{root0}run1/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_start_path = f'{root0}run1/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_start_path = f'{root0}run1/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径

    InD_train_dataset_end_path = f'{root}run1/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_end_path = f'{root}run1/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_end_path = f'{root}run1/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径

    # 使用np.load()函数读取npy文件
    InD_train_start_dataset = np.load(InD_train_dataset_start_path)
    InD_test_start_dataset  = np.load(InD_test_dataset_start_path)
    OOD_start_dataset       = np.load(OOD_dataset_start_path)

    InD_train_end_dataset   = np.load(InD_train_dataset_end_path)
    InD_test_end_dataset    = np.load(InD_test_dataset_end_path)
    OOD_end_dataset         = np.load(OOD_dataset_end_path)

    # # # 归一化
    InD_train_start_dataset = norm(InD_train_start_dataset)
    InD_test_start_dataset  = norm(InD_test_start_dataset)
    OOD_start_dataset       = norm(OOD_start_dataset)
    InD_train_end_dataset   = norm(InD_train_end_dataset)
    InD_test_end_dataset    = norm(InD_test_end_dataset)
    OOD_end_dataset         = norm(OOD_end_dataset)

    InD_train_dataset = np.concatenate((InD_train_start_dataset, InD_train_end_dataset), axis=1)
    InD_test_dataset = np.concatenate((InD_test_start_dataset, InD_test_end_dataset), axis=1)
    OOD_dataset = np.concatenate((OOD_start_dataset, OOD_end_dataset), axis=1)

    InD_similarity = cosine_similarity_matrix(InD_test_dataset, InD_train_dataset)
    OOD_similarity = cosine_similarity_matrix(OOD_dataset, InD_train_dataset)
    InD_concat_logits = 1 - np.max(InD_similarity, axis=1)
    OOD_concat_logits = 1 - np.max(OOD_similarity, axis=1)

    return InD_concat_logits, OOD_concat_logits

# for dataset in datasets:
#     for split in splits:
#         for pre in ['sup_vitb16_imagenet21k']:      #'mae_vitb16',
#                 root = f'/data/Jiuqing/Enhance_OOD_plant_new_split/Finetune_OOD/{dataset}{split}/{pre}/lr0.0_wd0.0/'
#                 root_0 = f'/data/Jiuqing/Enhance_OOD_plant_new_split/Linear_OOD/{dataset}{split}/{pre}/lr0.0_wd0.0/'
#                 print(f'{dataset}{split}    k-NN')
#
#                 InD_concat_logits, OOD_concat_logits = concat_kNN_OOD(root, root0, dataset)
#                 get_and_print_results(None, InD_concat_logits, OOD_concat_logits, auroc_list, aupr_list, fpr_list)
#                 InD_logits_sum, OOD_logits_sum = Diff_kNN_OOD(root, root_0, dataset)
#                 get_and_print_results(None, InD_logits_sum, OOD_logits_sum, auroc_list, aupr_list, fpr_list)

def kNN_OOD_Enhance(root,  root_0, name, method):
    ''' implementation of KNN '''
    InD_train_dataset_start_path = f'{root_0}run1/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_start_path = f'{root_0}run1/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_start_path = f'{root_0}run1/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径

    InD_train_dataset_end_path = f'{root}run1/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_end_path = f'{root}run1/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_end_path = f'{root}run1/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径

    # 使用np.load()函数读取npy文件
    InD_train_start_dataset = np.load(InD_train_dataset_start_path)
    InD_test_start_dataset  = np.load(InD_test_dataset_start_path)
    OOD_start_dataset       = np.load(OOD_dataset_start_path)

    InD_train_end_dataset   = np.load(InD_train_dataset_end_path)
    InD_test_end_dataset    = np.load(InD_test_dataset_end_path)
    OOD_end_dataset         = np.load(OOD_dataset_end_path)

    # # # 归一化
    InD_train_start_dataset = norm(InD_train_start_dataset)
    InD_test_start_dataset  = norm(InD_test_start_dataset)
    OOD_start_dataset       = norm(OOD_start_dataset)
    InD_train_end_dataset   = norm(InD_train_end_dataset)
    InD_test_end_dataset    = norm(InD_test_end_dataset)
    OOD_end_dataset         = norm(OOD_end_dataset)

    # # 只用start的特征，打开这两行
    # InD_similarity = cosine_similarity_matrix(InD_test_start_dataset, InD_train_start_dataset)
    # OOD_similarity = cosine_similarity_matrix(OOD_start_dataset, InD_train_start_dataset)

    # # 只用end的特征 打开这两行
    # InD_similarity = cosine_similarity_matrix(InD_test_end_dataset, InD_train_end_dataset)
    # OOD_similarity = cosine_similarity_matrix(OOD_end_dataset, InD_train_end_dataset)

    # start 和end 联合特征 ，打开这五行
    InD_train_dataset = np.concatenate((InD_train_start_dataset, InD_train_end_dataset), axis=1)
    InD_test_dataset = np.concatenate((InD_test_start_dataset, InD_test_end_dataset), axis=1)
    OOD_dataset = np.concatenate((OOD_start_dataset, OOD_end_dataset), axis=1)
    InD_similarity = cosine_similarity_matrix(InD_test_dataset, InD_train_dataset)
    OOD_similarity = cosine_similarity_matrix(OOD_dataset, InD_train_dataset)

    InD_concat_logits = 1 - np.max(InD_similarity, axis=1)
    OOD_concat_logits = 1 - np.max(OOD_similarity, axis=1)

    InD_score_path = f'{root}run1/epoch_0_test_{method}_score.npy'  # 替换为你的实际文件路径
    OOD_score_path = f'{root}run1/epoch_0_{name}_{method}_score.npy'  # 替换为你的实际文件路径
    InD_score = np.load(InD_score_path)
    OOD_score = np.load(OOD_score_path)

    InD_score, OOD_score = norm_score(InD_score, OOD_score)

    InD_enhance_logits = InD_concat_logits + InD_score
    OOD_enhance_logits = OOD_concat_logits + OOD_score

    return InD_enhance_logits, OOD_enhance_logits, InD_score, OOD_score

for dataset in datasets:
    for split in splits:
        for pre in ['imagenet22k_sup_rnx_base']: #'mae_vitb16', imagenet22k_sup_rnx_base, sup_vitb16_imagenet21k
            root = f'/data/Jiuqing/Enhance_OOD_plant_new_split/CNN_Finetune_OOD/{dataset}{split}/{pre}/lr0.0_wd0.0/'
            root_0 = f'/data/Jiuqing/Enhance_OOD_plant_new_split/CNN_Linear_OOD/{dataset}{split}/{pre}/lr0.0_wd0.0/'

            if not os.path.exists(root) or not os.path.exists(root_0):
                continue

            for method in methods:
                # print(f'{dataset}{split}    {method}')
                InD_concat_logits, OOD_concat_logits, InD_score, OOD_score = kNN_OOD_Enhance(root, root_0, dataset, method)
                get_and_print_results(None, InD_concat_logits, OOD_concat_logits, auroc_list, aupr_list, fpr_list)
                # get_and_print_results(None, InD_score, OOD_score, auroc_list, aupr_list, fpr_list)

            # print(f'{dataset}{split}    k-NN')
            InD_logits_sum, OOD_logits_sum, InD_similarity_end, OOD_similarity_end = Diff_kNN_OOD(root, root_0, dataset)
            get_and_print_results(None, InD_logits_sum, OOD_logits_sum, auroc_list, aupr_list, fpr_list)
            # et_and_print_results(None, InD_similarity_end, OOD_similarity_end, auroc_list, aupr_list, fpr_list)
