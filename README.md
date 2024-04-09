# CRetinex
Code of "CRetinex: A Progressive Color-shift Aware Retinex Model for Low-light Image Enhancement".<br>
This method can keep the color constancy of the low-light image (as can be seen from the enhancement results of the low-light images captured of the same scene).

<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/CRetinex_ex.png" width="870" height="168"/></div>
<br>

## Recommended Environment:
python=3.6<br>
tensorflow-gpu=1.14.0<br>
numpy=1.19<br>
scikit-image=0.17.2<br>
pillow=8.2<br>

### To train:
Download the training data: [LOL](https://daooshee.github.io/BMVC2018website/), [AGLIE](https://phi-ai.buaa.edu.cn/project/AgLLNet/index.htm), and [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark) datasets.

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
