# w207_project4

## Team ErrorBusters!

**Team members**: Alice Ye, Prakhar Maini, Simran Bhatia, Srishti Mehra 

**Project Link**: https://www.kaggle.com/c/random-acts-of-pizza/ 

**Presentation Link**: https://docs.google.com/presentation/d/19EtoVyiOs-Hx7zCe0ZpoviAotcX1UWjiZL5ZJL_FzSo/edit?usp=sharing

## Repository Guide
Plesae see the summary of cotent in each folder as below:

|Item Name| Item Type| Description | Key Importance|
|---|---|---|---|
|.ipynb_checkpoints| Folder | This folder contains the saved check points of jupyter notebook sessions|  None|
|alice| Folder | This folder is Alice's data playground and could be visited to see the exploration Alice did during the project| Alice's data exploration|
|baseline_presentation| Folder | This folder contains the data-prep and baseline RF model that was developed for the first check-point| First check-point review|
|data/random-acts-of-pizza| Folder | This folder contains all the data (training and holdout) for RAOP problem in json and zip format | Raw data folder|
|error_analysis| Folder | This folder contains the individual error analysis notebooks that the team worked on. This includes the file "RAOP_Combined_notebook" which contains the final full feature engineering flow| Full feature engineering and error analysis|
|final_model| This is the key folder that contains out final fully tuned ensemble modeling framework with full fledged training and validation pipeline. Also, it contains the "kaggle_submisson_predictions.csv" file which is our final outcome for submission onto Kaggle platform| **Main model** | **Please use for model review** |
|prakhar| Folder | This folder is Prakhar's data playground and could be visited to see the exploration Alice did during the project| Prakhar's data exploration|
|simran| Folder | This folder is Simran's data playground and could be visited to see the exploration Alice did during the project| Simran's data exploration|
|srishti| Folder | This folder is Srishti's data playground and could be visited to see the exploration Alice did during the project| Srishti's data exploration|

As mentioned in the above section, our final model resides in the **final_model** folder in the notebook named **RAOP_Combined_Notebook_Final.ipynb**. 

Originally we had 4040 rows of training data. We divided that in 90/10 ratio between training and validation. In terms of feature engineering, we did the following:

1. Simpale text and non-text features (e.g. length, number of posts, number of sub-reddits, posting time, punctuation usage etc.)
2. Topic probability matrix using Non-negative matrix factorization for unigarms (10 latent topics) and bi-grams (5 latent topics)
3. Manual category tagging and binary feature generation
4. Basic sentiment scores per post using NLTK
5. Doc2Vec representation for each post 
6. Detection to top 10 keywords associated with success and failure classes and binary vector creation incorporating word similarity using word2vec 

After complete feature engineering, we eneded up with **111** distinct features that we used in modeling phase.

Finally, we tuned 11 base models ('svm','knn','random forest','Extra Trees','XgBoost','ada boost','gbm','logistic regression','Naive Bayes','NN','bagging classifier') on the training set and used a GBM as a meta learner on the probability outcomes of base models as the final form of model stacking. 

### Results

|Model|Model type|ROC-AUC|
|---|---|---|
|SVM|base|0.5796|
|KNN|base|0.5847|
|Random Forest|base|0.6307|
|Extra Trees|base|0.6407|
|XGBoost|base|0.6268|
|Adaboost|base|0.6526|
|GBM|base|0.6608|
|LR|base|0.6288|
|Gaussian NB|base|0.6359|
|NN|base|0.6247|
|Bagging Classifier|base|0.6359|
|**GBM**|**meta learner**|**0.7612**|

Using this stacked ensemble for prediction, we recevied 0.66544 ROC-AUC on kaggle private leaderboard which converts to rank 125.

#### Code Reference
1. Basic NLTK processing: https://www.kaggle.com/alvations/basic-nlp-with-nltk

# Project Overview 
## Learning Goals
* Composite modeling - Secondary Ensembles
* Feature selection techniques
  * Information based techniques (MI, PMI, mRMR)
  * Regularization 
  * Feature elimination
* Text Analysis
  * Word2vec
  * Doc2vec
* Feature engineering
  * Text based features
  * Temporal features 
  * Ensembling different data types (Auto Encoders)

## Project Elements
* Write-ups (deck)
  * Preliminary 
  * Final 
* EDA - https://docs.google.com/spreadsheets/d/1yUHsnTsQrMdf9VGGXcoCZI05Wi_X6Gs4eh3_D4DtsJA/edit#gid=0
* Feature Engineering
* Modeling
* Model Evaluation
  * Deciding on the metric
  * Model Selection 

## Cadence
Thursday - 6 PM PST every week

## Reference
* Word vector training - https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795
* Mutual Information - 
  * Feature Selection - https://thuijskens.github.io/2017/10/07/feature-selection/
  * sklearn feature selection - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#r50b872b699c4-4
  * Github implementation - https://github.com/dwave-examples/mutual-information-feature-selection/blob/master/titanic.py
  * paper - https://dl.acm.org/doi/10.1145/2623330.2623611
* Model ensembles 
  * Theory - https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
  * example - https://www.toptal.com/machine-learning/ensemble-methods-machine-learning


