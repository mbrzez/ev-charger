# Readme

This is a sample Machine Learning project using `XGBClassifier` and `LogisticRegression` to predict EV charger user type (a classification problem) and `XGBRegressor` (a regression problem) to predict EV charging cost.

Due to the poor quality of the dataset, the model accuracy for both cases is very low (<35%). Therefore, this primarily an academic use case.

**Kaggle dataset**: [Electric Vehicle Charging Patterns](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns)

## Project Includes:
* Data cleaning and preparation for analysis  
* Scaling and normalizing the data  
* Defining train and test sets  
* Defining models and initial parameters  
* Model evaluation and plotting confusion matrices for classification  
* Calculating permutation importance  
* Calculating Shapley values  
* Hyperparameter fine-tuning  