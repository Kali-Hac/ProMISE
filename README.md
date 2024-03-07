![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![Tensorflow >=1.14.0](https://img.shields.io/badge/Tensorflow->=1.14.0-yellow.svg)
![Pytorch >=1.1.0](https://img.shields.io/badge/Pytorch->=1.1.0-green.svg)
![Faiss-gpu >= 1.6.3](https://img.shields.io/badge/Faiss->=1.6.3-orange.svg)

# Probabilistic Masked Spatial-Temporal Skeleton Reconstruction: General Skeleton Semantics Learning for Person Re-Identification

## Introduction
This is the implementation of ProMISE presented by "Probabilistic Masked Spatial-Temporal Skeleton Reconstruction: General Skeleton Semantics Learning for Person Re-Identification".

## Environment
- Python >= 3.5
- Tensorflow-gpu >= 1.14.0
- Pytorch >= 1.1.0
- Faiss-gpu >= 1.6.3 

Here we provide a configuration file to install the extra requirements (if needed):
```bash
conda install --file requirements.txt
```

**Note**: This file will not install tensorflow/tensorflow-gpu, faiss-gpu, pytroch/torch, please install them according to the cuda version of your graphic cards: [**Tensorflow**](https://www.tensorflow.org/install/pip), [**Pytorch**](https://pytorch.org/get-started/locally/). Take cuda 9.0 for example:
```bash
conda install faiss-gpu cuda90 -c pytorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
conda install tensorflow==1.14
conda install sklearn
```

## Datasets and Models
We provide three already **pre-processed datasets** (IAS, BIWI, KGBD) with various sequence lengths (**f=4/6/8/10/12**) [**here (pwd: 7je2)**](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg) and the **pre-trained models** [**here (pwd: kzys)**](https://pan.baidu.com/s/1hGeP-mymedbjgxsubFmyPQ). Since we report the average performance of our approach on all datasets, here the provided models may produce better results than the paper. <br/>


Please download the pre-processed datasets and model files while unzipping them to ``Datasets/`` and ``ReID_Models/`` folders in the current directory. <br/>

**Note**: The access to the Vislab Multi-view KS20 dataset and large-scale RGB-based gait dataset CASIA-B are available upon request. If you have signed the license agreement and been granted the right to use them, please email us with the signed agreement and we will share the complete pre-processed KS20 and CASIA-B data. The original datasets can be downloaded here: [IAS](http://robotics.dei.unipd.it/reid/index.php/downloads), [BIWI](http://robotics.dei.unipd.it/reid/index.php/downloads), [KGBD](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor), [KS20](http://vislab.isr.ist.utl.pt/datasets/#ks20), [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp). We also provide the ``Preprocess.py`` for directly transforming original datasets to the formated training and testing data. <br/> 

## Dataset Pre-Processing
To (1) extract 3D skeleton sequences of length **f=6** from original datasets and (2) process them in a unified format (``.npy``) for the model inputs, please simply run the following command: 
```bash
python Pre-process.py 6
```
**Note**: If you hope to preprocess manually (or *you can get the [already preprocessed data (pwd: 7je2)](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg)*), please frist download and unzip the original datasets to the current directory with following folder structure:
```bash
[Current Directory]
├─ BIWI
│    ├─ Testing
│    │    ├─ Still
│    │    └─ Walking
│    └─ Training
├─ IAS
│    ├─ TestingA
│    ├─ TestingB
│    └─ Training
├─ KGBD
│    └─ kinect gait raw dataset
└─ KS20
     ├─ frontal
     ├─ left_diagonal
     ├─ left_lateral
     ├─ right_diagonal
     └─ right_lateral
```
After dataset preprocessing, the auto-generated folder structure of datasets is as follows:
```bash
Datasets
├─ BIWI
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ Still
│      │    └─ Walking
│      └─ train_npy_data
├─ IAS
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ A
│      │    └─ B
│      └─ train_npy_data
├─ KGBD
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ gallery
│      │    └─ probe
│      └─ train_npy_data
└─ KS20
    └─ 6
      ├─ test_npy_data
      │    ├─ gallery
      │    └─ probe
      └─ train_npy_data
```
**Note**: KS20 data need first transforming ".mat" to ".txt". If you are interested in the complete preprocessing of KS20 and CASIA-B, please contact us and we will share. We recommend to directly download the preprocessed data [**here (pwd: 7je2)**](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg).

## Applying ProMISE to State-of-the-Art Models
Here we provide two sample codes for applying **ProMISE** to TranSG and SPC-MGR models. The adaptation of ProMISE to other models can be simply implemented based on the attached [pesudo code](https://github.com/Kali-Hac/ProMISE/tree/main/pseudo-code).

### Application to [TranSG](https://arxiv.org/abs/2303.06819)
To apply **ProMISE** to TranSG to obtain skeleton representations and validate their effectiveness on the person re-ID task on a specific dataset (probe), please simply run the following command:  

```bash
python TranSG-Pro.py --dataset KS20 --probe probe

# Default options: --dataset KS20 --probe probe --length 6  --gpu 0
# --dataset [IAS, KS20, BIWI, KGBD]
# --probe ['probe' (the only probe for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)]
# --alpha 0.5
# --prob_t 0.5
# --prob_s 0.5
# --GPC_lambda 0.5 (the lambda for fusing downstream objective (GPC in TranSG) and SSL objective (ProMISE))
# --length [4, 6, 8, 10] 
# --(H, n_heads, L_transfomer, seq_lambda, GPC_lambda, lr, etc.) using optimal settings of TranSG 
# --mode [Train (for training), Eval (for testing)]
# --gpu [0, 1, ...]

```
Please see ```TranSG-Pro.py``` for more details.

To print evaluation results (Top-1, Top-5, Top-10 Accuracy, mAP) of the best model saved in default directory (```ReID_Models/(Dataset)/(Probe)```), run:

```bash
python TranSG-Pro.py --dataset KS20 --probe probe --mode Eval
```

### Application to SPC-MGR
To apply **ProMISE** to SPC-MGR to obtain skeleton representations and validate their effectiveness on the person re-ID task on a specific dataset (probe), please simply run the following command:  

```bash
python SPC-MGR-Pro.py --dataset KS20 --probe probe

# Default options: --dataset KS20 --probe probe --length 6  --gpu 0
# --dataset [IAS, KS20, BIWI, KGBD, CASIA_B]
# --probe ['probe' (the only probe for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)] 
# --alpha 0.5
# --prob_t 0.5
# --prob_s 0.5
# --D_lambda 0.5 (the lambda for fusing downstream objective (SPC in SPC-MGR) and SSL objective (ProMISE))
# --length [4, 6, 8, 10] 
# --(t, lr, eps, min_samples, m, fusion_lambda)  using optimal settings of SPC-MGR
# --mode [UF (for unsupervised training), DG (for direct domain generalization)]
# --gpu [0, 1, ...]

```
Please see ```SPC-MGR-Pro.py``` for more details.


## Application to Model-Estimated Skeleton Data 

### Estimate 3D Skeletons from RGB-Based Scenes
To apply our approach to person re-ID under the large-scale RGB scenes (CASIA B), we exploit pose estimation methods to extract 3D skeletons from RGB videos of CASIA B as follows:
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Extract the 2D human body joints by using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Step 3: Estimate the 3D human body joints by using [3DHumanPose](https://github.com/flyawaychase/3DHumanPose)


We provide already pre-processed skeleton data of CASIA B for **single-condition** (Nm-Nm, Cl-Cl, Bg-Bg) and **cross-condition evaluation** (Cl-Nm, Bg-Nm) (**f=40/50/60**) [**here (pwd: 07id)**](https://pan.baidu.com/s/1_Licrunki68r7F3EWQwYng). 
Please download the pre-processed datasets into the directory ``Datasets/``. <br/>

### Usage
To (1) apply **ProMISE** to TranSG to obtain skeleton representations and (2) validate their effectiveness on the person re-ID task on CASIA B under **single-condition** and **cross-condition** settings, please simply run the following command:

```bash
python TranSG-Pro.py --dataset CAISA_B --probe_type nm.nm --length 40

# --length [40, 50, 60] 
# --probe_type ['nm.nm' (for 'Nm' probe and 'Nm' gallery), 'cl.cl', 'bg.bg', 'cl.nm' (for 'Cl' probe and 'Nm' gallery), 'bg.nm']
# --alpha 0.5
# --prob_t 0.5
# --prob_s 0.5
# --GPC_lambda 0.5 (the lambda for fusing downstream objective (GPC in TranSG) and SSL objective (ProMISE))
# --(H, n_heads, L_transfomer, seq_lambda, prompt_lambda, GPC_lambda, lr, etc.) with default settings
# --gpu [0, 1, ...]

```


## Application to Different Skeleton Modeling

### Usage
To (1) apply **ProMISE** to TranSG on skeleton representations of a specific modeling level (joint-level, part-level, body-level) and (2) individually validate the effectiveness on the person re-ID task on the person re-ID task on a specific dataset (probe), please simply run the following command:  

```bash
python TranSG-Pro.py --dataset KS20 --probe probe --level J

# Default options: --dataset KS20 --probe probe --length 6
# --dataset [IAS, KS20, BIWI, KGBD]
# --probe ['probe' (the only probe for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)]
# --level [J (joint-level), P (part-level), B (body-level)]
```

## [Pesudo Codes for ProMISE](https://github.com/Kali-Hac/ProMISE/tree/main/pseudo-code)

## Future Update
We are collecting and integrating all state-of-the-art skeleton-based models. A model zoo with user-friendly APIs will be released for researchers to seamlessly apply ProMISE and other skeleton semantics learning (SSL) tasks for evaluation.

## License

ProMISE is released under the MIT License. Our models and codes must only be used for the purpose of research.
