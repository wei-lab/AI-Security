# Type-I Genrative Adversarial Attack

ReColorAdv
This is an implementation of the GAA adversarial attack and other attacks described in the IJCAI 2021 paper "Type-I Genrative Adversarial Attack".

## Getting Started
Clone this repository by running

git clone https://github.com/wei-lab/AI-Security/Type-I Genrative Adversarial Attack

You can experiment with the GAA attack, by itself and combined with other attacks, in the getting_started.ipynb Jupyter notebook. You can also open the notebook in Google Colab via the badge below.

Open In Colab

You can also install the GAA package with pip by running

pip install recoloradv 

## Evaluation Script (CIFAR-10)
The script evaluate_cifar10.py will evaluate a model trained on CIFAR-10 against the adversarial attacks in Table 1 of the paper. For instance, to evaluate a CIFAR-10 model trained on delta (L-infinity) attacks against a ReColorAdv+delta attack, run

python Type-I Genrative Adversarial Attack/CIFAR10/train_gan.py 

## Evaluation Script MNIST

python Type-I Genrative Adversarial Attack/Mnist/train_gan.py 
