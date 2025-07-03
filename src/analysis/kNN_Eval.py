import numpy as np
from src.analysis.tsne import plot_tsne


def kNN_OOD(root, epoch, name):
    ''' implementation of KNN '''
    if epoch != 0:
        epoch = 'best'
    InD_train_dataset_path = f'{root}/epoch_{epoch}_InD_train_feature.npy'  # 替换为你的实际文件路径
    InD_test_dataset_path = f'{root}/epoch_{epoch}_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_path = f'{root}/epoch_{epoch}_OOD_{name}_feature.npy'  # 替换为你的实际文件路径

    InD_train_dataset = np.load(InD_train_dataset_path)
    InD_test_dataset = np.load(InD_test_dataset_path)
    OOD_dataset = np.load(OOD_dataset_path)

    InD_similarity = cosine_similarity_matrix(InD_test_dataset, InD_train_dataset)
    OOD_similarity = cosine_similarity_matrix(OOD_dataset, InD_train_dataset)

    InD_logits = 1 - np.max(InD_similarity, axis=1)
    OOD_logits = 1 - np.max(OOD_similarity, axis=1)

    plot_tsne(root, epoch, name)
    return InD_logits, OOD_logits


def norm_kNN_OOD(root, epoch, name):
    ''' OOD Evaluation by using vector norm '''
    if epoch != 0:
        epoch = 'best'
    InD_test_dataset_path = f'{root}/epoch_{epoch}_InD_test_feature.npy'  # 替换为你的实际文件路径
    OOD_dataset_path = f'{root}/epoch_{epoch}_OOD_{name}_feature.npy'  # 替换为你的实际文件路径

    InD_test_dataset = np.load(InD_test_dataset_path)
    OOD_dataset = np.load(OOD_dataset_path)

    InD_test_norms = np.linalg.norm(InD_test_dataset, axis=1)
    OOD_norms = np.linalg.norm(OOD_dataset, axis=1)

    return InD_test_norms, OOD_norms


def norm(matrix1):
    return matrix1 / np.linalg.norm(matrix1, axis=1, keepdims=True)     # 归一化


def cosine_similarity_matrix(matrix1, matrix2):
    normalized_matrix1 = norm(matrix1)
    normalized_matrix2 = norm(matrix2)

    similarity_matrix = np.dot(normalized_matrix1, normalized_matrix2.T)
    # similarity_matrix = np.dot(matrix1, matrix2.T)

    return similarity_matrix
