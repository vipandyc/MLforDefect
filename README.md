# Machine learning for defect
Code repository for solving defects with machine learning. (Under active development; new features to be added)

## Installation guide
This code relies on PyTorch and machine learning interatomic potentials (MLIPs). For a smooth construction of a compatible python environment, we recommend users to follow the steps below:

1. **(Recommended)** Create a conda virtual environment. Specifying `python=3.12` version works for our case.
```
conda create -n crygen
conda activate crygen
``` 
2. Install the torch version (recommended with CUDA GPU) 2.4.1. Please check the cuda version most compatible to your machine on the official torch website ([torch website](https://pytorch.org/get-started/previous-versions/)). **Important: please specify `"numpy<2"` in the command.** For example run:
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 "numpy<2" -c pytorch -c nvidia
```
3. Now we install mace first. Please follow the official documentation of mace ([MACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html)), since it's still under active development.
4. **(Optional)** Then we install mattersim. If you followed the recommended steps above, simply running `pip install` works. For more info please check the official documentation of MatterSim ([MatterSim](https://github.com/microsoft/mattersim)). For example run:
```
pip install mattersim
```
