# CRetinex
Code of "CRetinex: A Progressive Color-shift Aware Retinex Model for Low-light Image Enhancement".<br><br>
This method can keep the color constancy of the low-light image (as can be seen from the enhancement results of the low-light images captured of the same scene).
<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/CRetinex_ex.png" width="870" height="168"/></div>
<br>

The framework of this method is shown below:
<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/CRetinex_framework.png" width="870" height="348"/></div>
<br>


## Recommended Environment:
python=3.6<br>
tensorflow-gpu=1.14.0<br>
numpy=1.19<br>
scikit-image=0.17.2<br>
pillow=8.2<br>

### To train:
* __Training dataset:__
  *  Download the training data: [LOL](https://daooshee.github.io/BMVC2018website/), [AGLIE](https://phi-ai.buaa.edu.cn/project/AgLLNet/index.htm), and [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark) datasets.
  * Select part of the data for training, and put the low-light images and corresponding normal-light images in `./dataset/low/` and `./dataset/high/`, respectively.
  * Can also put a small number of paired low-light and normal-light images in `./dataset/eval/low/` and `./dataset/eval/high/` for validation during the training phase.

* __Train the decomposition network:__<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_decomposition_network.py```<br>
  * The relevant files are stored in `./checkpoint/decom/`, `./logs/decom/`, and `./train_decom_result/`

* __Train the color shift estimation network:__<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_color_network.py```<br>
  * The relevant files are stored in `./checkpoint/color_net/`, `./logs/color_net/`, and `./train_color_result/`

* Train the spatially variant pollution estimation network:<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_noise_network.py```<br>
  * The relevant files are stored in `./checkpoint/noise_net/`, `./logs/noise_net/`, and `./train_noise_result/`

* __Train the illumination adjustment network:__<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_illumination_adjustment_network.py```<br>
  * The relevant files are stored in `./checkpoint/illumination_adjustment/`, `./logs/illumination_adjustment/`, and `./train_illumination_adjust_result/`

### To test:

If this work is helpful to you, please cite it as:
```
@article{xu2024CRetinex,
  title={CRetinex: A Progressive Color-shift Aware Retinex Model for Low-light Image Enhancement},
  author={Xu, Han and Zhang, Hao and Yi, Xunpeng and Ma, Jiayi},
  journal={International Journal of Computer Vision},
  year={2024},
  publisher={Springer}
}
```
