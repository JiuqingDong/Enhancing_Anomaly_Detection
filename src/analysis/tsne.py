import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import seaborn as sns

def plot_tsne(root, epoch, name):
    InD_train_feature = np.load(f'{root}/epoch_{epoch}_InD_train_feature.npy')
    InD_test_feature = np.load(f'{root}/epoch_{epoch}_InD_test_feature.npy')
    OOD_feature = np.load(f'{root}/epoch_{epoch}_OOD_{name}_feature.npy')

    # 设置 sample_size 为这三个数组中最小的长度
    sample_size = min(InD_train_feature.shape[0], InD_test_feature.shape[0], OOD_feature.shape[0])

    sample_indices = np.random.choice(InD_train_feature.shape[0], sample_size, replace=False)
    InD_train_feature_ = InD_train_feature[sample_indices]
    sample_indices = np.random.choice(InD_test_feature.shape[0], sample_size, replace=False)
    InD_test_feature_ = InD_test_feature[sample_indices]
    sample_indices = np.random.choice(OOD_feature.shape[0], sample_size, replace=False)
    OOD_feature_ = OOD_feature[sample_indices]

    # 计算欧几里得范数
    norms_InD_train = np.linalg.norm(InD_train_feature_, axis=1)
    norms_InD_test = np.linalg.norm(InD_test_feature_, axis=1)
    norms_OOD = np.linalg.norm(OOD_feature_, axis=1)

    InD_train_feature_ = InD_train_feature_ / norms_InD_train[:, np.newaxis]
    InD_test_feature_ = InD_test_feature_ / norms_InD_test[:, np.newaxis]
    OOD_feature_ = OOD_feature_ / norms_OOD[:, np.newaxis]

    # 合并抽样数据和标签
    combined_data = np.vstack((InD_train_feature_, InD_test_feature_, OOD_feature_))
    labels = np.repeat(['InD Train', 'InD Test', 'OOD Test'], sample_size)

    # 为每个数据组设置颜色
    colors = ['blue', 'orange', 'green']

    # 创建t-SNE对象
    tsne = TSNE(n_components=2, random_state=42)

    # 对合并数据进行降维
    data_tsne = tsne.fit_transform(combined_data)

    # 绘制t-SNE图，使用不同颜色表示不同数据组
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(data_tsne[labels == label, 0],
                    data_tsne[labels == label, 1], color=colors[i], label=label, marker='.', s=50, alpha=0.7)    # s 大小

    plt.title('t-SNE Visualization of Three Data Groups')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 添加图例
    plt.legend(loc='upper right')

    # 创建目录（文件夹）并保存t-SNE图
    output_directory = f'{root}/tsne/'
    os.makedirs(output_directory, exist_ok=True)
    plt.savefig(os.path.join(output_directory, f'epoch_{epoch}_tsne_plot_{name}.png'), dpi=300)

    # plot_dist(norms_InD_train, norms_InD_test, norms_OOD, epoch, root, name)


def plot_dist(norms_InD_train, norms_InD_test, norms_OOD, epoch, root, name):
    min_value = np.minimum.reduce([norms_InD_train, norms_InD_test, norms_OOD])
    max_value = np.maximum.reduce([norms_InD_train, norms_InD_test, norms_OOD])
    gap = max_value - min_value

    train_norm = (norms_InD_train - min_value) / gap
    test_norm = (norms_InD_test - min_value) / gap
    ood_norm = (norms_OOD - min_value) / gap

    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83', '#FF8C00']

    sns.displot({"ID_Train": train_norm, "ID_Test": test_norm, "OOD": ood_norm}, label="id", kind="kde", palette=palette, fill=True, alpha=0.4)

    output_directory = f'{root}/tsne/'
    os.makedirs(output_directory, exist_ok=True)
    plt.savefig(os.path.join(output_directory, f'epoch_{epoch}_{name}_feature_norm.png'), dpi=300)
