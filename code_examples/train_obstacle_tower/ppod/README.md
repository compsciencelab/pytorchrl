## Guided Exploration with Proximal Policy Optimization using a Single Demonstration

Code example of the PPO+D algorithm. The paper can be found [here](https://arxiv.org/pdf/2007.03328.pdf).

```
@inproceedings{libardi2021guided,
  title={Guided Exploration with Proximal Policy Optimization using a Single Demonstration},
  author={Libardi, Gabriele and De Fabritiis, Gianni and Dittert, Sebastian},
  booktitle={International Conference on Machine Learning},
  pages={6611--6620},
  year={2021},
  organization={PMLR}
}
```

### 0. Configuration file

All parameters can be adjusted at:

    ./code_examples/train_animalai/ppod/conf.yaml

### 1. Record Demonstrations

Run the following script and control the agent with keyboard keys "W, A, S, D". 

    ./code_examples/train_animalai/ppod/record_demo.sh

### 2. Train Agent
    
    ./code_examples/train_animalai/ppod/train.sh

### 3. Visualize Demonstrations

    ./code_examples/train_animalai/ppod/visualize_demos.sh

### 4. Enjoy Agent Performance

    ./code_examples/train_animalai/ppod/enjoy.sh