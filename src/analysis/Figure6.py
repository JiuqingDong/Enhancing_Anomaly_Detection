from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns
from src.engine.evaluator import get_and_print_results


def cosine_similarity_matrix(matrix1, matrix2):
    return np.dot(matrix1, matrix2.T)


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

    return ind_scores_norm, ood_scores_norm


def kNN_OOD_Enhance(root, name, root_0, method):
    ''' implementation of KNN '''
    InD_train_dataset_start_path = f'{root_0}run1/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_start_path = f'{root_0}run1/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_start_path = f'{root_0}run1/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径
    InD_train_start_dataset = np.load(InD_train_dataset_start_path)
    InD_test_start_dataset  = np.load(InD_test_dataset_start_path)
    OOD_start_dataset       = np.load(OOD_dataset_start_path)

    InD_train_dataset_end_path = f'{root}run2/epoch_0_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_end_path = f'{root}run2/epoch_0_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_end_path = f'{root}run2/epoch_0_OOD_{name}_feature.npy'  # 替换为你的实际文件路径
    InD_train_end_dataset   = np.load(InD_train_dataset_end_path)
    InD_test_end_dataset    = np.load(InD_test_dataset_end_path)
    OOD_end_dataset         = np.load(OOD_dataset_end_path)

    # # # # # 归一化
    InD_train_start_dataset = norm(InD_train_start_dataset)
    InD_test_start_dataset  = norm(InD_test_start_dataset)
    OOD_start_dataset       = norm(OOD_start_dataset)

    InD_train_end_dataset   = norm(InD_train_end_dataset)
    InD_test_end_dataset    = norm(InD_test_end_dataset)
    OOD_end_dataset         = norm(OOD_end_dataset)

    # 只用start的特征，打开这两行
    InD_similarity_start = cosine_similarity_matrix(InD_test_start_dataset, InD_train_start_dataset)
    OOD_similarity_start = cosine_similarity_matrix(OOD_start_dataset, InD_train_start_dataset)

    # 只用end的特征 打开这两行
    InD_similarity_end = cosine_similarity_matrix(InD_test_end_dataset, InD_train_end_dataset)
    OOD_similarity_end = cosine_similarity_matrix(OOD_end_dataset, InD_train_end_dataset)

    # # start 和end 联合特征 ，打开这五行
    # InD_train_dataset = np.concatenate((InD_train_start_dataset, InD_train_end_dataset), axis=1)
    # InD_test_dataset = np.concatenate((InD_test_start_dataset, InD_test_end_dataset), axis=1)
    # OOD_dataset = np.concatenate((OOD_start_dataset, OOD_end_dataset), axis=1)
    # InD_similarity = cosine_similarity_matrix(InD_test_dataset, InD_train_dataset)
    # OOD_similarity = cosine_similarity_matrix(OOD_dataset, InD_train_dataset)

    InD_concat_logits_start = 1 - np.max(InD_similarity_start, axis=1)
    OOD_concat_logits_start = 1 - np.max(OOD_similarity_start, axis=1)

    InD_concat_logits_end = 1 - np.max(InD_similarity_end, axis=1)
    OOD_concat_logits_end = 1 - np.max(OOD_similarity_end, axis=1)

    InD_score = np.load(f'{root}run1/epoch_0_test_{method}_score.npy')
    OOD_score = np.load(f'{root}run1/epoch_0_{name}_{method}_score.npy')
    # 归一化
    InD_score, OOD_score = norm_score(InD_score, OOD_score)
    InD_concat_logits_start, OOD_concat_logits_start = norm_score(InD_concat_logits_start, OOD_concat_logits_start)
    InD_concat_logits_end, OOD_concat_logits_end = norm_score(InD_concat_logits_end, OOD_concat_logits_end)

    InD_enhance_GK = (InD_score + InD_concat_logits_start)/2
    OOD_enhance_GK = (OOD_score + OOD_concat_logits_start)/2

    InD_enhance_GK_DSK = (InD_score + InD_concat_logits_start + InD_concat_logits_end)/3
    OOD_enhance_GK_DSK = (OOD_score + OOD_concat_logits_start + OOD_concat_logits_end)/3

    return InD_score, OOD_score, InD_enhance_GK, OOD_enhance_GK, InD_enhance_GK_DSK, OOD_enhance_GK_DSK



def plot_distribution(id_scores, ood_scores, knownledge, method, output_path='/data/Jiuqing/Enhance_OOD_plant_new_split/Figure'):
    min_value = np.minimum(id_scores.min(), ood_scores.min())
    max_value = np.maximum(id_scores.max(), ood_scores.max())
    gap = max_value - min_value
    # 将数据归一化
    id_scores_normalized = (id_scores - min_value) / gap
    ood_scores_normalized = (ood_scores - min_value) / gap

    sample_size = min(len(id_scores_normalized), len(ood_scores_normalized))
    id_scores_normalized = np.random.choice(id_scores_normalized, sample_size, replace=False)
    ood_scores_normalized = np.random.choice(ood_scores_normalized, sample_size, replace=False)

    # 设置样式和调色板
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']

    # 绘制归一化的分布图
    sns.displot({"Known": id_scores_normalized, "Unknown": ood_scores_normalized}, label="id", kind="kde", palette=palette, fill=True, alpha=0.8)
    plt.xlim(-0.1, 1.1)

    # 保存图像
    if knownledge == None:
        plt.savefig(os.path.join(f"{output_path}/ori_{method}.png"), bbox_inches='tight', dpi=200)
    else:
        plt.savefig(os.path.join(f"{output_path}/ori_{method}_{knownledge}.png"), bbox_inches='tight', dpi=200)

auroc_list, aupr_list, fpr_list = [], [], []
for dataset in ['plant_village',]:
    for split in ['_1', ]:
        for pre in ['sup_vitb16_imagenet21k', ]:#'mae_vitb16',
            root = f'/data/Jiuqing/Enhance_OOD_plant_new_split/Prompt_OOD/{dataset}{split}/{pre}/lr0.0_wd0.0/'
            root_0 = f'/data/Jiuqing/Enhance_OOD_plant_new_split/Linear_OOD/{dataset}{split}/{pre}/lr0.0_wd0.0/'
            if not os.path.exists(root) or not os.path.exists(root_0):
                continue
            for method in ['max_logits', ]:
                print(f'{dataset}{split}   {method}')
                InD_score, OOD_score, InD_enhance_GK, OOD_enhance_GK, InD_enhance_GK_DSK, OOD_enhance_GK_DSK, = kNN_OOD_Enhance(root, dataset, root_0, method)
                get_and_print_results(None, InD_score, OOD_score, auroc_list, aupr_list, fpr_list)
                get_and_print_results(None, InD_enhance_GK, OOD_enhance_GK, auroc_list, aupr_list, fpr_list)
                get_and_print_results(None, InD_enhance_GK_DSK, OOD_enhance_GK_DSK, auroc_list, aupr_list, fpr_list)

                plot_distribution(InD_score, OOD_score, None, method)
                plot_distribution(InD_enhance_GK, OOD_enhance_GK, 'GK', method)
                plot_distribution(InD_enhance_GK_DSK, OOD_enhance_GK_DSK, 'GK_DSK', method)
