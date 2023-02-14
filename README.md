# GAST
This is our Tensorflow implementation for our GAST 2023 paper and a part of baselines:

>Bin Wu, Lihong Zhong & Yangdong Ye. Graph-augmented social translation model for next-item recommendation, IEEE Transactions on Industrial Informatics (TII), Accept

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.14.0
* numpy == 1.16.4
* scipy == 1.3.1
* pandas == 0.17

## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.
NOTE: The cpp implementation is much faster than python.**

## Examples to run GCARec:
run [main.py](./main.py) in IDE or with command line:
```
python main.py
```

NOTE :   
(1) the duration of training and testing depends on the running environment.  
(2) set model hyperparameters on .\conf\GAST.properties  
(3) set NeuRec parameters on .\NeuRec.properties  
(4) the log file save at .\log\Gowalla_yiding_u5_s3\  

## Dataset
We provide Gowalla_yiding_u5_s3(Gowalla) dataset.
  * .\dataset\Gowalla_yiding_u5_s3.rating and Gowalla_yiding_u5_s3.uu
  *  Each line is a user with her/his positive interactions with items: userID \ itemID \ ratings \time.
  *  Each user has more than 10 associated actions.

## Baselines
The list of available models in GAST, along with their paper citations, are shown below:

| General Recommender | Paper                                                                                                         |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| BPRMF               | Steffen Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.                   |
| LightGCN            | Xiangnan He, et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.|
| SGL                 | J. Wu, X. Wang, F. Feng, X. He, L. Chen, J. Lian, and X. Xie. Self supervised graph learning for recommendation. SIGIR, 2021.|

| Sequential Recommender | Paper                                                                                                      |
|---------------------|---------------------------------------------------------------------------------------------------------------|
|SASRec          |W. Kang and J. J. McAuley. Self-attentive sequential recommendation. ICDM, 2018.|
|TransRec        |R. He, W. Kang, and J. McAuley. Translation-based recommendation. RecSys, 2017.|

| Social Recommender | Paper                                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------------|
| EATNN              | C. Chen, M. Zhang, C. Wang, W. Ma, M. Li, Y. Liu, and S. Ma. An efficient adaptive transfer neural network for social-aware recommendation. SIGIR, 2019.|
| EAGCN              | B. Wu, L. Zhong, L. Yao, and Y. Ye. EAGCN: An efficient adaptive graph convolutional network for item recommendation in social internet of things, IOT, 2022.|
