# WaveNet
- This repository is used to store the source code of model WaveNet. 
- And our Net is simple and easy to use.
- You can set the `argparse` in `training.py` to make a choice with the baselines and datasets. And our source code also support download the datasets and preprocess them with decomposing as well. Therefore, you can use the following simple script to run our model.
`
python training.py
`
If you want to repeat the others datasets performance, it need to decompose them again and cost a few time, please wait with patience.

Last, our code main reference is [NIPS 2021 BernNet](https://github.com/ivam-he/BernNet). And we fix the low pyg version bug when loading datasets, which makes our repsitory easier to use.

Thanks for your considering!
