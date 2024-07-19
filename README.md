## Intersecting Boundary Sensitive Fingerprinting (IBSF)
![Python 3.6.13](https://img.shields.io/badge/python-3.6.13-green.svg?style=plastic)
![PyTorch 1.10.1](https://img.shields.io/badge/torch-1.10.1-green.svg?style=plastic)

This repository contains the PyTorch implementation of the following paper to appear at ICML 2024:

> **Intersecting Boundary Sensitive Fingerprinting for Tampering Detection of DNN Models**<br>
> Xiaofan Bai, Chaoxiang He, Xiaojing Ma, Bin Benjamin Zhu, and Hai Jin  


  
## Quick Start
First, install all dependencies via pip.
```shell
$ pip install -r requirements.txt
```

## Make Output dir
Second, prepare dir to save fingerprints
```shell
$ mkdir outputs
$ cd outputs
$ mkdir fingerprints
$ cd ../..
```

### IBSF fingerpirnt samples generation
Below is a demo to generate 1000 fingerprint samples on GPU device  
```shell
$ python mian.py --gpu '1' --num 1000 --dataset 'cifar10'
```
This saves the generated fingerprints file at ``outputs/fingerprints/cifar10``.

## Datasets
Our IBSF currently implements custom data loaders for the following datasets. 

- CIFAR-10 
- ImageNet (needs manual download)
- GTSRB (needs manual download)







