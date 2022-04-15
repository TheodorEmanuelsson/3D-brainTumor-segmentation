# 3D Multimodal Brain Tumor Segmentation

## The Data

The data is supplied by [CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html) at Perelman School of Medicine, University of Pennsylvania and is avaiable on [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data). The data are multimodal scans and are given as .nii.gz files. The datset contains 4 volumes but in this project one three volumes are used: the post-contrast T1-weighted (T1Gd), T2-weighted (T2) and Fluid Attenuated Inversion Recovery (T2-FLAIR).

Slice of a brain volume:
![slice](https://user-images.githubusercontent.com/49917684/163586729-82e52cbe-f964-4051-b286-bea9a6d9bed8.png)

## Preprocessing

Preprocessing steps include Min-Max-Scaling, converting nii files to numpy arrays and cropping.

For the purposes of this project, the volume is cropped into the following format:

![cropped](https://user-images.githubusercontent.com/49917684/163586822-c75ed149-8535-4541-8842-7431141489d5.png)

With the accompanying segmentation mask an image can look as follows:

![mask](https://user-images.githubusercontent.com/49917684/163586922-d6d2fb9e-b437-41d9-9246-fb382ac7f9d8.png)


## Training

Training a simple [Unet](https://arxiv.org/pdf/1505.04597.pdf) model adapted to the 3D setting, consisting of 5,645,828 parameters.

![trainloss](https://user-images.githubusercontent.com/49917684/163587252-8238d776-ede4-4523-8d63-4258d642168f.png)

![trainaccuracy](https://user-images.githubusercontent.com/49917684/163587263-ec749396-e3f7-406d-939c-a6d88664c679.png)

## Performance

For a batch of 4 training volumes, the model achieves a mean IoU of 0.82.

Prediction results:
![download](https://user-images.githubusercontent.com/49917684/163587347-65c2aaf7-b64f-4e61-a64e-c230e470dae7.png)




