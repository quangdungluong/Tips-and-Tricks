# Boost CV LB with RAPIDS SVR
* A "post process" trick. After we have trained NN model, we can freeze the backbone and train another RAPIDS SVR (support vector regression) head using the extracted embeddings. Then during inference, we predict with both the original NN head and the new RAPIDS SVR head. We average the two predictions.

* This has the advantage of using BE classification loss with the first NN head. And using RMSE regression loss with the second RAPIDS SVR head. Therefore our final solution uses both classification and regression loss.

## How to add RAPIDS SVR Head
* There are 3 steps to build a double headed model. The first step is to train NN backbone and head. The next step is to freeze the NN backbone and train a RAPIDS SVR head with extracted embeddings. Lastly, infer with both heads and average the predictions.

![step1](/assets/svr_step1.png)
![step2](/assets/svr_step2.png)
![step3](/assets/svr_step3.png)

## Starter Notebook
* [RAPIDS SVR](https://www.kaggle.com/cdeotte/rapids-svr-boost-17-8)
