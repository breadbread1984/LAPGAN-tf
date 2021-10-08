# LAPGAN

this project implements algorithm LAPGAN for high resolution image generation introduced in paper [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/abs/1506.05751).

## install prerequisite packages

install with command

```shell
pip3 install -r requirements.txt
```

## prepare dataset

download and prepare dataset with the following command

```shell
python3 create_dataset.py
```

test the downloaded dataset with the command

```shell
python3 create_dataset.py --test
```

## how to train


