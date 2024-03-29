# WaveNet: Tackling Non-stationary Graph Signals via Graph Spectral Wavelets
- This repository is used to store the source code of model WaveNet. 
- Our paper is available in https://ojs.aaai.org/index.php/AAAI/article/view/28781
- And our Net is simple and easy to use. You can set the `argparse` in `training.py` to make a choice with the baselines and datasets. And our source code also support download the datasets and preprocess them with decomposing as well. 

### Environment 
```
pip install -r requirements.txt
```

### Running the code
```
python training.py --dataset Chameleon --K 30
```


### Citation
```
@article{he2021bernnet,
  title={WaveNet: Tackling Non-stationary Graph Signals via Graph Spectral Wavelets},
  author={Yang, Zhirui and Hu, Yulan and Ouyang, Sheng and Liu, Jingyu and Wang, Shuqiang and Ma, Xibo and Wang, Wenhan and Su, Hanjing and Liu, Yong},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
