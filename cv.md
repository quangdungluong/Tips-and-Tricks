There are 2 main approaches.
* Train N Folds models and ensemble the N test predictions.
* After discovering the best hyperparameters from CV, train 1 model using 100% data and infer test with 1 model.

In approach one, let's say you have 5-fold models. Then you split the train data into 5 folds and train 5 models. Each model trains with 80% data and infers the other 20% data. When we combine all the 5x 20% train predictions, we have 1 prediction for each train frame. We compute our CV score on this (called OOF). During inference each of the 5 fold models predicts the test data. Then we have 5 predictions for each test frame. We ensemble these with WBFT.

In approach two, we start with 5-fold models. After finding the best hyperparameters, we train another model that uses 100% train data. Afterward we use this 1 model to predict the test data. We have 1 prediction for each test frame. This is our submission.

[Selection OOF Ensemble](https://www.kaggle.com/cdeotte/forward-selection-oof-ensemble-0-942-private/notebook)