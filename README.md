# TransProto
Zhuang Chen, Tieyun Qian: "[Retrieve-and-Edit Domain Adaptation for End2End Aspect Based Sentiment Analysis](https://ieeexplore.ieee.org/abstract/document/9693267)". Accepted by TASLP 2022.

Data and code will be released soon.

# 2022.08.24 UPDATE

Sorry for the late update.

In the past half year, I have be occupied with my Ph.D. dissertation. Thus the code release is postponed.

The code and data are retrieved from my backups. If you have any problems when running, please feel free to raise an issue.



## 1. Requirements
 To reproduce the reported results accurately, please install the specific version of each package.

* python 3.6.7
* pytorch 1.5.0
* pytorch-pretrained-bert 0.4.0
* numpy 1.19.1

## 2. Usage
 We incorporate the training and evaluation of TransProto in the **all_bert_bridge.sh**. Just run it as below.

```
CUDA_VISIBLE_DEVICES=0 bash all_bert_bridge.sh
```


## 3. Pre-trained BERT
* Download [bert-cross](https://drive.google.com/file/d/1M9XJctC4aYcAs7jlBgpXMqRoQtqOVhUR/view?usp=sharing), and unzip it in the folder **bert**.


## 4. Citation
If you find our code and datasets useful, please cite our paper.

  
```
@article{DBLP:journals/taslp/ChenQ22,
  author    = {Zhuang Chen and
               Tieyun Qian},
  title     = {Retrieve-and-Edit Domain Adaptation for End2End Aspect Based Sentiment
               Analysis},
  journal   = {{IEEE} {ACM} Trans. Audio Speech Lang. Process.},
  volume    = {30},
  pages     = {659--672},
  year      = {2022},
  url       = {https://doi.org/10.1109/TASLP.2022.3146052},
  doi       = {10.1109/TASLP.2022.3146052},
  timestamp = {Wed, 23 Feb 2022 11:17:51 +0100},
  biburl    = {https://dblp.org/rec/journals/taslp/ChenQ22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

:checkered_flag: 
