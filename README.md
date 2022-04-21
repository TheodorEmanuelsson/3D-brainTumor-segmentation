# 3D Multimodal Brain Tumor Segmentation

## The Data

The data is supplied by [CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html) at Perelman School of Medicine, University of Pennsylvania and is avaiable on [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data). The data are multimodal scans and are given as .nii.gz files. The datset contains 4 volumes but in this project one three volumes are used: the post-contrast T1-weighted (T1Gd), T2-weighted (T2) and Fluid Attenuated Inversion Recovery (T2-FLAIR).

Slice of a brain volume:
![slice](https://user-images.githubusercontent.com/49917684/163586729-82e52cbe-f964-4051-b286-bea9a6d9bed8.png)

## Preprocessing

Preprocessing steps include Min-Max-Scaling, converting nii files to numpy arrays and cropping.

For the purposes of this project, the volume is cropped into the following format:

![cropped](https://user-images.githubusercontent.com/49917684/163586822-c75ed149-8535-4541-8842-7431141489d5.png)

After scaling:

![mask](https://user-images.githubusercontent.com/49917684/163586922-d6d2fb9e-b437-41d9-9246-fb382ac7f9d8.png)


## Training

Training a simple [Unet](https://arxiv.org/pdf/1505.04597.pdf) model adapted to the 3D setting, consisting of 5,645,828 parameters and is trained for 100 epochs.

![trainlossV1](https://user-images.githubusercontent.com/49917684/164469594-be17aa6f-b768-4115-ba69-104d800a9011.png)

![trainaccV1](https://user-images.githubusercontent.com/49917684/164469627-18d9a062-d8c3-4184-b97a-147dd6a3f6b4.png)


A 3D Unet with attention blocks is also trained for 100 epochs.

![trainlossAtt](https://user-images.githubusercontent.com/49917684/164469673-48c9908e-9ed8-4934-b6ee-e9910a914729.png)

![trainaccAtt](https://user-images.githubusercontent.com/49917684/164469710-6e1ab798-e143-4a06-ad80-518e067449b8.png)


## Performance

Comparing the mean IoU on the test data for the two models. It seems that the attention model is slightly better compared to the base model.

Basic 3D Unet: 0.7957
Attention 3D Unet: 0.8339

![pred](https://user-images.githubusercontent.com/49917684/164469966-f9e27ccc-7fce-4f08-b134-6269ff6ea4df.png)


