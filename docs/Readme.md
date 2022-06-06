# **Model Overview**

A pre-trained model for volumetric (3D) segmentation of brain tumors from MRIs based on BraTS 2018 data.

## Workflow
The model is trained to annotate the "tumor core" (TC) subregion of primary (gliomas) brain tumors based on 1 input MRI scans (T1c) and 6 extreme points. The TC describes the bulk of the tumor, which is what is typically resected. The annotation algorithm is described in [1]. The network architecture is described in [2]. The workflow is described as below.

![workflow](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_tumors_annotation_t1ce_tc_workflow.png)

## Data

The training data is from the Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018.
- Target: TC tumor subregion
- Task: Annotation
- Modality: MRI
- Size: 285 3D volumes

The provided labeled data was partitioned, based on our own split, into training (200 studies), validation (42 studies) and testing (43 studies) datasets.

# **Training configuration**

This model utilized a similar approach described in 3D MRI brain tumor segmentation using autoencoder regularization, which was a winning method in BraTS2018 [1]. The training was performed with the following:

- Script: train.sh
- GPU: At least 16GB of GPU memory.
- Actual Model Input: not fixed
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: DiceLoss

## Input

**Input:** 2 channel 
- channel 1: MRI (T1c at 1x1x1 mm)
- channel 2: Gaussian heatmap using extreme points

**Preprocessing:**
1. Crop foreground of image using ground truth label with margin = 20
2. Zero-pad cropped foreground image to make it Divisible by 32
3. Randomly spatial flipping
4. Normalizing to unit std with zero mean
5. Randomly scaling and shifting intensity of the volume
6. Add channel of Gaussian heatmap using extreme points

## Output

**Output:** 1 channel TC tumor subregion

# **Model Performance**
The model was trained with 200 cases with our own split, as shown in the datalist json file in config folder. The achieved Dice scores on the validation and testing data are:

Tumor core (TC): 0.862 (testing)

## Training and Validation Performance

![train](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_tumors_annotation_t1ce_tc_train.png)

![val](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_tumors_annotation_t1ce_tc_val.png)

# **Intended Use**

The model needs to be used with NVIDIA hardware and software. For hardware, the model can run on any NVIDIA GPU with memory greater or equal to 16 GB. For software, this model is usable only as part of Transfer Learning & Annotation Tools in Clara Train SDK container. Find out more about Clara Train at the Clara Train Collections on NGC.

**The Clara pre-trained models are for developmental purposes only and cannot be used directly for clinical procedures.**

# **License**

[End User License Agreement](https://developer.download.nvidia.com/assets/Clara/Downloads/Secure/NVIDIA%20Clara%20SDK%20License_13_Mar_2020.pdf?ZLZeEolG2cxcdF7CIwVpxixla_K5D25Ssy8ZO-6UaNEFYgu3QYBBS1N__qsZDVASZckVS2zHqUuaGDJmJ8TSu15LYpyU1sljzLD3qwbQt7p3FChwVCjl4DkC3aiE3miR7K07OX06QNQVkjZtdMp87XJJN5wWQkGh7Vo7nulDXOjGRAd8LIMNOa15LQ) is included with the product. Licenses are also available along with the model application zip file. By pulling and using the Clara Train SDK container and downloading models, you accept the terms and conditions of these licenses.

# **References**

**[1] Maninis, Kevis-Kokitsi, et al. "Deep extreme cut: From extreme points to object segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018. [https://arxiv.org/abs/1711.09081](https://arxiv.org/abs/1711.09081).** 

**[2] Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." International MICCAI Brainlesion Workshop. Springer, Cham, 2018. [https://arxiv.org/pdf/1810.11654.pdf](https://arxiv.org/pdf/1810.11654.pdf).**