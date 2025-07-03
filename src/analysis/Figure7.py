import numpy as np
import json

root = '/Users/jiuqingdong/Desktop/PlantDetPaper/Towards few-shot out-of-distribution detection/Imagenet1k_2shot_3_Score'

# zero_score_path = f'{root}/without_FT_Imagenet_2shot_iNaturalist_kNN_score_score.npy'  # 替换为你的实际文件路径
# FFT_score_path  = f'{root}/FFT_Imagenet_2shot_iNaturalist_kNN_score_score.npy'  # 替换为你的实际文件路径
# FFT_DSGF_score_path= f'{root}/FFT_DSGF_Imagenet_2shot_iNaturalist_kNN_score_score.npy'  # 替换为你的实际文件路径
# # 使用np.load()函数读取npy文件
# zero_score = np.load(zero_score_path)
# FFT_score = np.load(FFT_score_path)
# FFT_DSGF_score = np.load(FFT_DSGF_score_path)
#
# # 打开 JSON 文件
# with open(f'{root}/ood_iNaturalist.json', 'r') as file:
#     # 从文件中加载 JSON 数据并解析成字典
#     ood_iNaturalist = json.load(file)
#
# # 现在，变量 data 包含了 JSON 文件中的字典数据
# iNaturalist_list = list(ood_iNaturalist.keys())
#
# for i in range(len(zero_score)):
#     zero_score_i = zero_score[i]
#     FFT_score_i = FFT_score[i]
#     FFT_DSGF_score_i = FFT_DSGF_score[i]
#
#     if zero_score_i > FFT_score_i and FFT_DSGF_score_i > FFT_score_i and FFT_DSGF_score_i > zero_score_i:
#         diff_1 = zero_score_i - FFT_score_i
#         diff_2 = FFT_DSGF_score_i - FFT_score_i
#         if diff_1 > 0.03 and diff_2 > 0.03 and FFT_score_i < 0.65:
#             print(iNaturalist_list[i].replace('ood/iNaturalist/images/', ''), zero_score_i, FFT_score_i, FFT_DSGF_score_i)
#
#     # if  FFT_score_i > zero_score_i and FFT_DSGF_score_i > zero_score_i and FFT_DSGF_score_i > FFT_score_i:
#     #     diff_1 = FFT_score_i - zero_score_i
#     #     diff_2 = FFT_DSGF_score_i - FFT_score_i
#     #     if diff_1 > 0.2 and diff_2 > 0.03:
#     #         print(iNaturalist_list[i].replace('ood/iNaturalist/images/', ''), zero_score_i, FFT_score_i, FFT_DSGF_score_i)




zero_score_path = f'{root}/without_FT_Imagenet_2shot_ID_test_kNN_score_score.npy'  # 替换为你的实际文件路径
FFT_score_path  = f'{root}/FFT_Imagenet_2shot_ID_test_kNN_score_score.npy'  # 替换为你的实际文件路径
FFT_DSGF_score_path= f'{root}/FFT_DSGF_Imagenet_2shot_ID_test_kNN_score_score.npy'  # 替换为你的实际文件路径
# 使用np.load()函数读取npy文件
zero_score = np.load(zero_score_path)
FFT_score = np.load(FFT_score_path)
FFT_DSGF_score = np.load(FFT_DSGF_score_path)

# 打开 JSON 文件
with open(f'{root}/id_imagenet1k_test.json', 'r') as file:
    # 从文件中加载 JSON 数据并解析成字典
    ood_iNaturalist = json.load(file)

# 现在，变量 data 包含了 JSON 文件中的字典数据
iNaturalist_list = list(ood_iNaturalist.keys())

for i in range(len(zero_score)):
    zero_score_i = zero_score[i]
    FFT_score_i = FFT_score[i]
    FFT_DSGF_score_i = FFT_DSGF_score[i]

    if FFT_score_i < zero_score_i and FFT_DSGF_score_i < zero_score_i:
        diff_1 = zero_score_i - FFT_score_i
        if diff_1 > 0.1 and FFT_score_i < 0.2:
            print(iNaturalist_list[i], zero_score_i, FFT_score_i, FFT_DSGF_score_i)
#
    # if FFT_score_i > zero_score_i and FFT_score_i > FFT_DSGF_score_i:
    #     # diff_1 = FFT_score_i - zero_score_i
    #     diff_2 = FFT_score_i - FFT_DSGF_score_i
    #     if diff_2 > 0.09 and FFT_score_i < 0.5:
    #         print(iNaturalist_list[i].replace('ood/iNaturalist/images/', ''), zero_score_i, FFT_score_i, FFT_DSGF_score_i)

    # if  zero_score_i > FFT_score_i:
    #     diff_2 = FFT_score_i - FFT_DSGF_score_i
    #     if -0.05<diff_2<0 and FFT_score_i<0.2:
    #         print(iNaturalist_list[i], zero_score_i, FFT_score_i, FFT_DSGF_score_i)