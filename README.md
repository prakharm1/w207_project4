# w207_project4

## Team ErrorBusters!

**Team members**: Alice Ye, Prakhar Maini, Simran Bhatia, Srishti Mehra 

**Project Link**: https://www.kaggle.com/c/random-acts-of-pizza/ 

**Presentation Link**: https://docs.google.com/presentation/d/19EtoVyiOs-Hx7zCe0ZpoviAotcX1UWjiZL5ZJL_FzSo/edit?usp=sharing

**Project Goal**: Predict whether a post in [Random Acts of Pizza](https://www.reddit.com/r/Random_Acts_Of_Pizza/) will receive a pizza

## Architecture
![Comparison of ROC AUC by models](/img/architecture.png) 

## Project Components
1. **Feature Engineering**:
We started off with 4040 rows of training data which we divided in a 90/10 ratio between training and validation. We added features to our training and evaluation set using feature engineering:

- Simple text and non-text features (e.g. length, number of posts, number of sub-reddits, posting time, punctuation usage etc.)
- Topic probability matrix using Non-negative matrix factorization for unigarms (10 latent topics) and bi-grams (5 latent topics)
- Manual category tagging and binary feature generation
- Basic sentiment scores per post using NLTK
- Doc2Vec representation for each post 
- Detection to top 10 keywords associated with success and failure classes and binary vector creation incorporating word similarity using word2vec 

After complete feature engineering, we eneded up with **111** distinct features that we used in modeling phase.

2. **Building and Tuning Models**:
We built and hyperparameter-tuned 11 base models trained on our training set 
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Extra Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
- [Neural Network](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Bagging Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) 


3. **Ensemble Model Building**:
We used a [Stacked Ensemble](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) model to the take the predicted probabilities of the (above mentioned) 11 models as inputs and learn how to best combine their predictions for a final output prediction. We used a hyperparameter-tuned Gradient Boosting to be able for this learning.

4. **Results**:
![Comparison of ROC AUC by models](/img/final_roc_auc_curves.png)

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

The prediction from the Stacked ensemble had 0.66544 ROC-AUC on kaggle private leaderboard which converts to rank 125.


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
|img| Folder | This folder contains the images used in the README| Images |
|prakhar| Folder | This folder is Prakhar's data playground and could be visited to see the exploration Prakhar did during the project| Prakhar's data exploration|
|simran| Folder | This folder is Simran's data playground and could be visited to see the exploration Simran did during the project| Simran's data exploration|
|srishti| Folder | This folder is Srishti's data playground and could be visited to see the exploration Srishti did during the project| Srishti's data exploration|

As mentioned in the above section, our final model resides in the **final_model** folder in the notebook named **RAOP_Combined_Notebook_Final.ipynb**. 

## Project Learning Goals
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

## Cadence
Thursday - 6 PM PST every week

## References
* Word vector training - https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795
* Mutual Information - 
  * Feature Selection - https://thuijskens.github.io/2017/10/07/feature-selection/
  * sklearn feature selection - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#r50b872b699c4-4
  * Github implementation - https://github.com/dwave-examples/mutual-information-feature-selection/blob/master/titanic.py
  * paper - https://dl.acm.org/doi/10.1145/2623330.2623611
* Model ensembles 
  * Theory - https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
  * example - https://www.toptal.com/machine-learning/ensemble-methods-machine-learning
* Basic NLTK processing: https://www.kaggle.com/alvations/basic-nlp-with-nltk

