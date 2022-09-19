# habitat2021

## Installation

```
conda create -n habitat2021 python=3.7
source activate habitat2021
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install git+https://github.com/PyTorchRL/baselines.git
pip install pytorchrl
conda install habitat-sim headless -c conda-forge -c aihabitat -y
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all
conda install ipython
```