Light Field Video Frame Interpolation Using Hierarchical Spatial-Angular Temporal Information Decoupling and Fusion

<br>

**This is the PyTorch implementation of the Light Field Video Frame Interpolation in our paper
"Light Field Video Frame Interpolation Using Hierarchical Spatial-Angular Temporal Information Decoupling and Fusion". 

Network Architecture:
<img width="1734" height="659" alt="image" src="https://github.com/user-attachments/assets/37b48d10-7f08-486c-a1b7-bba8dd3477cf" />

Codes and Models:
Implementation details: The experiments involved in this paper are conducted using the Pytorch framework, with detailed 
versions of Python 3.10.18 and PyTorch 2.7.1+cu128. The experimental environment is configured as an AMD Ryzen 9 9950X 
16-Core CPU, 96GB RAM, and an NVIDIA RTX 5090 GPU (32GB).
Matlab (For test data generation and performance evaluation)

Datasets:LFV-Raytrix  and LFV-Lytro

L. Guillo, X. Jiang, G. Lafruit, C. Guillemot, Light field video dataset captured by a R8 Raytrix camera, ISO/IEC JTC1/SC29/WG11 Technical Report, 2018. 
T. C. Wang, J. Y. Zhu, N. K. Kalantari, A. A. Efros, R. Ramamoorthi, Light field video capture using a learning-based hybrid imaging system, ACM Trans. Graph. 36 (4) (2017) 1-13. 

Datasets: The LF data used in this experiment all come from two publicly available LFV datasets (i.e., LFV-Raytrix , 
LFV-Lytro ), both of which are established by capturing real-world scenes. These two datasets contain motions of varying 
degrees and directions (horizontal, vertical, and rotation). Specifically, the LFV-Raytrix dataset  is 
acquired by a Raytrix R8 LF camera. It contains 3 scenes, each comprising 300 frames, with each LF frame having a spatial 
resolution of 1920×1080 and an angular resolution of 5×5. The LFV-Lytro dataset is acquired by a Lytro lllum LF camera. 
It contains 150 scenes, each comprising 9 frames, with each LF frame having a spatial resolution of 541×376 and an angular 
resolution of 8×8.

In this work, we use 120 scenes from the LFV-Lytro dataset for training and the remaining 30 scenes for testing. In 
addition, all 3 scenes of the LFV-Raytrix dataset  are used for testing. Therefore, the number of training scenes is 120, and 
the number of test scenes is 33, which do not overlap with each other. For unified processing, we first extract the first 9 frames 
of LF images from the two datasets as the base data and crop their angular resolution to 5×5. Subsequently, the first, third, and 
fifth frames and the fifth, seventh, and ninth frames are used as two sets of samples, with the third and seventh frames as the 
ground truth of the two sets of samples. Ultimately, this yields 240 sets of training samples and 66 sets of test samples.


Train:
* Run `train.py` to perform network training.
* Checkpoint will be saved to `./Checkpoints/`.

Test:
* Run `test.py` to perform network inference.
The PSNR, SSIM, Y-PSNR,Y-SSIM,LFIQA,values of each dataset will be ./metrics to Matlab evaluation
