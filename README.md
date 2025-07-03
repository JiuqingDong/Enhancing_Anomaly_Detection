# Enhancing Anomaly Detection in Plant Disease Recognition with Knowledge Ensemble

* This code is an implementation of our manuscript: Enhancing Anomaly Detection in Plant Disease Recognition with Knowledge Ensemble.

* Authors: Jiuqing Dong, et al.
* We will release the complete code and complete this documentation after the article is published.


## Installation

Please check `env_setup.sh`.


## Pre-trained model preparation

Download and place the pre-trained Transformer-based backbones to './models/'. In our study, we use the [ViT-Base pre-trained](https://drive.google.com/file/d/11KuAkntDTPPcq4h4JwSjbDebNgVkgceA/view?usp=drive_link) on Imagenet-21k.

## Dataset Prepairation

  Cotton disease dataset: [https://www.kaggle.com/datasets/dhamur/cotton-plant-disease](https://www.kaggle.com/datasets/dhamur/cotton-plant-disease)
  
  Mango disease dataset: [https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)
  
  Strawberry disease dataset: [https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)
  
  Tomato disease dataset and Plant village dataset: [https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw)

please split the dataset by using our code.

## Train

  sh train.sh



## We will release the complete code and complete this documentation after our manuscript is accepted.

## How to use this code for a customer dataset?






