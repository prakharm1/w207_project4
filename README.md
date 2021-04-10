# w207_project4

## Team ErrorBusters!

**Team members**: Alice Ye, Prakhar Maini, Simran Bhatia, Srishti Mehra 

**Project Link**: https://www.kaggle.com/c/random-acts-of-pizza/ 

## Repository Guide
Plesae see the summary of cotent in each folder as below:

|Item Name| Item Type| Description | Key Importance|
|---|---|---|---|
|.ipynb_checkpoints| Folder | This folder contains the saved check points of jupyter notebook sessions|  None|
|alice| Folder | This folder is Alice's data playground and could be visited to see the exploration Alice did during the project| Alice's data exploration|
|baseline_presentation| Folder | This folder contains the data-prep and baseline RF model that was developed for the first check-point| First check-point review|
|data/random-acts-of-pizza| Folder | This folder contains all the data (training and holdout) for RAOP problem in json and zip format | Raw data folder|
|error_analysis| Folder | This folder contains the individual error analysis notebooks that the team worked on. This includes the file "RAOP_Combined_notebook" which contains the final full feature engineering flow| Full feature engineering and error analysis|
|final_model| This is the key folder that contains out final fully tuned ensemble modeling framework with full fledged training and validation pipeline. Also, it contains the "kaggle_submisson_predictions.csv" file which is our final outcome for submission onto Kaggle platform| **Main model** |
|prakhar| Folder | This folder is Prakhar's data playground and could be visited to see the exploration Alice did during the project| Prakhar's data exploration|
|simran| Folder | This folder is Simran's data playground and could be visited to see the exploration Alice did during the project| Simran's data exploration|
|sristi| Folder | This folder is Srishti's data playground and could be visited to see the exploration Alice did during the project| Srishti's data exploration|


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


