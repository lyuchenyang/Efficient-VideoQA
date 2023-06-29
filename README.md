<div align="center">

# Is a Video worth $n\times n$ Images? A Highly Efficient Approach to Transformer-based Video Question Answering 💡

**[Chenyang Lyu](https://lyuchenyang.github.io), [Tianbo Ji](mailto:jitianbo@ntu.edu.cn), [Yvette Graham](mailto:ygraham@tcd.ie), [Jennifer Foster](mailto:jennifer.foster@dcu.ie)**

School of Computing, Dublin City University, Dublin, Ireland 🏫

</div>

This repository contains the code for the Efficient-VideoQA system, which is a highly efficient approach for Transformer-based Video Question Answering. The system utilizes existing vision-language pre-trained models and converts video frames into a $n\times n$ matrix, reducing the computational requirements while maintaining the temporal structure of the original video.

## Table of Contents

- [1. Introduction](#1-introduction-📚)
- [2. Dataset](#2-dataset-📊)
- [3. Pre-processing](#3-pre-processing-🔧)
- [4. Training](#4-training-🎓)
- [5. Usage](#5-usage-🚀)
- [6. Dependencies](#6-dependencies-⚙️)

## 1. Introduction 📚

Conventional Transformer-based Video Question Answering (VideoQA) approaches generally encode frames independently through one or more image encoders followed by interaction between frames and questions. However, such approach incurs significant memory usage and inevitably slows down the training and inference speed. In this work, we present a highly efficient approach for VideoQA based on existing vision-language pre-trained models. We concatenate video frames into a $n\times n$ matrix and then convert it into one image. By doing so, we reduce the use of the image encoder from $n^{2}$ to $1$ while maintaining the temporal structure of the original video.

## 2. Dataset 📊

Please download the dataset from this link: [https://www.mediafire.com/folder/h14iarbs62e7p/shared](https://www.mediafire.com/folder/h14iarbs62e7p/shared) including videos and corresponding annotations. Move them under the `data/` directory.

Please download the TrafficQA dataset from this link: [https://sutdcv.github.io/SUTD-TrafficQA/#/download](https://sutdcv.github.io/SUTD-TrafficQA/#/download) including videos and corresponding annotations. Move them under the `data/` directory.

## 3. Pre-processing 🔧

To pre-process the data, use `data_preprocess.py` to extract and combine frames from videos in the MSR-VTT and TrafficQA dataset. Then tokenize the annotation data to tensor dataset.

## 4. Training 🎓

To train the model, use the following scripts:

- For TrafficQA dataset: `python run_trafficqa_concat_image.py --do_train --do_eval --num_train_epochs 2 --learning_rate 5e-6 --train_batch_size 8 --eval_batch_size 16 --attention_heads 8 --eval_steps 50`
- For MSR-VTT dataset: `python run_msrvtt_concat_image.py --do_train --do_eval --num_train_epochs 3 --learning_rate 5e-6 --train_batch_size 16 --eval_batch_size 16 --attention_heads 8 --eval_steps 5000`

## 5. Usage 🚀

Once the model is trained, you can use it for VideoQA tasks. Provide a video, and the system will give the most probable answer based on the video. 🔎

## 6. Dependencies ⚙️

Make sure to install the following dependencies before running the code:

- Python (>=3.8) 🐍
- PyTorch (>=2.0) 🔥
- MoviePy 🧮
- ffmpeg 🐼

## Citation 📄

If you find our paper useful, please cite it using the bibtex below:

```bibtex
@article{lyu2023video,
  title={Is a Video worth $ n$\backslash$times n $ Images? A Highly Efficient Approach to Transformer-based Video Question Answering},
  author={Lyu, Chenyang and Ji, Tianbo and Graham, Yvette and Foster, Jennifer},
  journal={arXiv preprint arXiv:2305.09107},
  year={2023}
}
```
