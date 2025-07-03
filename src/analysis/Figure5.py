import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def plot_tsne(root, n):
    # 加载三组数据
    PT_InD_train_feature = np.load(f'{root}/PT_InD_train_feature.npy')  # [50000, 768]
    PT_InD_test_feature = np.load(f'{root}/PT_InD_test_feature.npy')    # [50000, 768]
    PT_OOD_feature = np.load(f'{root}/PT_OOD_iNaturalist_feature.npy')  # [10000, 768]

    FT_InD_train_feature = np.load(f'{root}/FT_InD_train_feature.npy')  # [50000, 768]
    FT_InD_test_feature = np.load(f'{root}/FT_InD_test_feature.npy')    # [50000, 768]
    FT_OOD_feature = np.load(f'{root}/FT_OOD_iNaturalist_feature.npy')  # [10000, 768]

    # 计算欧几里得范数
    norms_PT_InD_train = np.linalg.norm(PT_InD_train_feature, axis=1)
    norms_PT_InD_test = np.linalg.norm(PT_InD_test_feature, axis=1)
    norms_PT_OOD = np.linalg.norm(PT_OOD_feature, axis=1)
    norms_FT_InD_train = np.linalg.norm(FT_InD_train_feature, axis=1)
    norms_FT_InD_test = np.linalg.norm(FT_InD_test_feature, axis=1)
    norms_FT_OOD = np.linalg.norm(FT_OOD_feature, axis=1)

    # 正则化为单位向量
    PT_InD_train_feature_normalized = PT_InD_train_feature / norms_PT_InD_train[:, np.newaxis]
    PT_InD_test_feature_normalized = PT_InD_test_feature / norms_PT_InD_test[:, np.newaxis]
    PT_OOD_feature_normalized = PT_OOD_feature / norms_PT_OOD[:, np.newaxis]
    FT_InD_train_feature_normalized = FT_InD_train_feature / norms_FT_InD_train[:, np.newaxis]
    FT_InD_test_feature_normalized = FT_InD_test_feature / norms_FT_InD_test[:, np.newaxis]
    FT_OOD_feature_normalized = FT_OOD_feature / norms_FT_OOD[:, np.newaxis]

    # concat
    combined_InD_train_features_normalized = np.hstack((PT_InD_train_feature_normalized, FT_InD_train_feature_normalized))
    combined_InD_test_features_normalized = np.hstack((PT_InD_test_feature_normalized, FT_InD_test_feature_normalized))
    combined_OOD_features_normalized = np.hstack((PT_OOD_feature_normalized, FT_OOD_feature_normalized))

    # 随机抽取每组数据的400个样本
    sample_size = min(1000, len(combined_InD_train_features_normalized))

    sample_indices = np.random.choice(combined_InD_train_features_normalized.shape[0], sample_size, replace=False)
    combined_InD_train_features_normalized_ = combined_InD_train_features_normalized[sample_indices]
    sample_indices = np.random.choice(combined_InD_test_features_normalized.shape[0], sample_size, replace=False)
    combined_InD_test_features_normalized_ = combined_InD_test_features_normalized[sample_indices]
    sample_indices = np.random.choice(combined_OOD_features_normalized.shape[0], sample_size, replace=False)
    combined_OOD_features_normalized_ = combined_OOD_features_normalized[sample_indices]

    # 合并抽样数据和标签
    combined_data = np.vstack((combined_InD_train_features_normalized_, combined_InD_test_features_normalized_, combined_OOD_features_normalized_))
    labels = np.repeat(['ID Train Samples', 'ID Test Samples', 'OOD Samples'], sample_size)

    # 为每个数据组设置颜色
    colors = ['blue', 'orange', 'green']
    sizes = [70, 70, 100]
    alphas = [0.3, 0.5, 0.75]
    # 创建t-SNE对象
    tsne = TSNE(n_components=2, random_state=42)

    # 对合并数据进行降维
    data_tsne = tsne.fit_transform(combined_data)

    # 绘制t-SNE图，使用不同颜色表示不同数据组
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(data_tsne[labels == label, 0],
                    data_tsne[labels == label, 1], color=colors[i], label=label, marker='.', s=sizes[i], alpha=alphas[i])    # s 大小


    # 添加图例
    plt.legend([]) # plt.legend(loc='upper right')
    plt.axis('off')
    # 绘制图的边框
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')  # 可以设置边框颜色为黑色或其他颜色

    # 创建目录（文件夹）并保存t-SNE图
    output_directory = f'{root}/tsne/'
    os.makedirs(output_directory, exist_ok=True)
    plt.savefig(os.path.join(output_directory, f'tsne2_{n}_plot_iNaturalist.png'), dpi=300)

root = '/Users/jiuqingdong/Desktop/PlantDetPaper/Towards few-shot out-of-distribution detection/Figure5_TSNE/FFT_OOD/Imagenet_1k_2shot'
plot_tsne(root, '0')