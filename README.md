BalancedFixMatch
========

BalancedFixMatch is the model to classify COVID-19 based on Chest X-ray images.

## Requirements

* Python 3.7
* PyTorch 1.7.0

## Datasets

### COVIDx-CXR Dataset ([Link](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md))

| Type | COVID-19 | Pneumonia | Normal | Total |
|:-:|:-:|:-:|:-:|:-:|
| Train | 517 | 5,475 | 7,966 | 13,958 |
| Test | 100 | 100 | 100 | 300 |

## Performances

| Labels | μ | Baseline | FixMatch | FixMatch&Focal loss |
|:-:|:-:|:-:|:-:|:-:|
| 150 | 1 | 85.25±1.97 | 85.03±1.08 | 84.94±1.33 |
| 150 | 2 | 85.25±1.97 | 85.81±1.21 | 86.00±1.46 |
| 150 | 3 | 85.25±1.97 | 85.92±1.04 | 86.25±1.69 |
| 300 | 1 | 87.06±1.30 | 87.39±1.41 | 86.86±1.15 |
| 300 | 2 | 87.06±1.30 | 87.25±1.24 | 87.67±1.28 |
| 300 | 3 | 87.06±1.30 | 87.56±1.01 | 87.78±0.96 |

## How to Train BalancedFixMatch

These are tutorials for training BalancedFixMatch using CXR images:

* [Train BalancedFixMatch](train.ipynb)

