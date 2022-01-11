# PPYOLO-Tiny for Raspberry 4B

## PPYOLO-Tiny

PP-YOLO Tiny is more suitable for mobile devices, which implement MobileNetV3 and Depthwise Seqarable Convolution.

The structure of PP-YOLO Tiny is shown at the end of this file.

## Paddle Lite

paddlepaddle is a deep learning framework, but it do not support Arm devices such as Raspberry. So, the official support another tool called [paddle-lite](https://github.com/PaddlePaddle/Paddle-Lite), which can be deployed on much more mobile devices.

## Usage

### About

This project is based on a detection mission which will be deployed on Raspberry 4B. The dataset is originally from [berkeley-BDD100K dataset](https://bdd-data.berkeley.edu/). The official annotations are two json files and I translate them into 79000 xml file (see [bdd100k_voc_labels](https://github.com/Feng1909/bdd100k_voc_labels)), so that we can easily train our model by using [PaddleX](https://github.com/PaddlePaddle/PaddleX).

### Requirement

```
# Window/Linux
python 				3.7
CUDA(Optional) 		11.0
cudnn(Optional) 	8.0
paddlepaddle-gpu    2.1.2.post110
paddlex             2.0.0rc3
```

```
# Raspberry OS 		
raspios_full_armhf 2020-05-28-05:28
# https://mirrors.tuna.tsinghua.edu.cn/raspberry-pi-os-images/raspios_full_armhf/archive/
```

### Step

1. Build **Paddle Lite** and install it **or install the package I have already built**

   The package is in the same respository called `paddlelite-a0e14603f-cp37-cp37m-linux_armv7l.whl`

   using `pip3 install ` can install paddle-lite.

   OR:

   Strongly suggest this blog [【超详细】树莓派4B 安装Paddle-Lite 2.8.0](https://blog.csdn.net/weixin_40973138/article/details/114780090?spm=1001.2014.3001.5501)

   **note: **When build paddle lite, use `./lite/tools/build_linux.sh --arch=armv7hf --with_python=ON --with_extra=ON --python_version=3.7 --with_cv=ON` **instead of** `./lite/tools/build_linux.sh --arch=armv7hf --with_python=ON`

2. Export your model

   By using paddlex, `paddlex --export_inference --model_dir=./output/ppyoloTiny/best_model --save_dir=./inference_model` is very simple.

3. Optimize your model

   This step can only be done on Raspberry, because the paddle-list version on PC may be different from Raspberry and the final version of building is a strange string (mine is a0e14603f)

   Using `opt.py` can translate model into paddle-lite form.

4. Run

   run `python3 predict.py` in your terminal is OK

   possible result:

   <img src='./result.jpg'>

   The model is trained from 1% of  BDD100K dataset for 250 epochs
   
   If you want to predict from a camera, change [if True](https://github.com/Feng1909/PPYOLO-Tiny/blob/main/predict.py#L7) into `if False`

## Structure of PPYOLO-Tiny

<img src="./model.pdmodel.png">