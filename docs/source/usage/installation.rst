Installation
============

Installing Anaconda or Miniconda
--------------------------------

If not already done, install conda (Miniconda is sufficient). To do so, see the `official documentation. <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_

Installing Nappo library
------------------------

1. Set up conda environment ::

    conda create -y -n nappo
    conda activate nappo

2. Install dependencies ::

    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    pip install git+git://github.com/openai/baselines.git

3. Install package ::

    pip install nappo
