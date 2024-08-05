# FedEgo: Privacy-preserving Personalized Federated Graph Learning with ego-graphs

Code and models from the paper [FedEgo: Privacy-preserving Personalized Federated Graph Learning with ego-graphs](https://arxiv.org/abs/2208.13685). 

# Usage

1. Download the data in the file_path according to `experiments.conf`
2. Use `python main.py` to run the code.
3. Or you can use `run.sh` to run the code.

# Note

1. Note that the parameters are fixed in the `main.py`, please comment out the part in the  for your own setting.
2. Please refer `options.py` for parameters setting in details.
3. If you use the script `run.sh` to run the code, please create new folders according to `run.sh`.

# Citation
Please cite us if our work is useful for your research.
```
@article{zhang2023fedego,
  title={FedEgo: privacy-preserving personalized federated graph learning with ego-graphs},
  author={Zhang, Taolin and Mai, Chengyuan and Chang, Yaomin and Chen, Chuan and Shu, Lin and Zheng, Zibin},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={18},
  number={2},
  pages={1--27},
  year={2023},
  publisher={ACM New York, NY}
}
```
