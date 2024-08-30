import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import mlflow
import argparse

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn_features.transformers import DataFrameSelector

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

FILE_PATH = os.path.join(os.getcwd(), '..', 'dataset.csv')
df = pd.read_csv(FILE_PATH)

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
# Handle Outliers

df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)

# Split to X & y

X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']

# Split to train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, stratify=y, random_state=45)

#data processing 

# Slice lists
num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
categ_cols = ['Geography', 'Gender']
ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))

# Pipeline

# Numerical: num_cols -> Impute using median, and Standardization
# Categorical: categ_cols -> Impute most_frequent, and OHE
# Ready: ready_cols -> Impute most_frequent


num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

categ_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(drop='first', sparse_output=False))
        ])

ready_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])


# Combine all
all_pipeline = ColumnTransformer(transformers=[
        ('numerical', num_pipeline, num_cols),
        ('categorical', categ_pipeline, categ_cols),
        ('ready', ready_pipeline, ready_cols)
    ])

# fit & transform
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)


#dealing with imbalanced daatset 
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)
dict_weight = {}
for i in range(2):
    dict_weight[i] = vals_count[i]


def train_model(X_train, y_train,plot_name:str,n_estimators:int,max_depth:int,class_weight=None ):
    
    
    mlflow.set_experiment('churn')
    with mlflow.start_run() as run:
        
        #setting tags
        mlflow.set_tag('clf','forest')
        
        #Classifier
        
        clf= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=45,class_weight=class_weight)
        clf.fit(X_train,y_train)
        
        #predict on test 
        y_pred_test = clf.predict(X_test_final)
        
        #metrics
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_test =f1_score(y_test, y_pred_test)
        
        #logging for metrics, params, model
        mlflow.log_metrics({'accuracy':accuracy_test,'f2_score':f1_test})
        mlflow.log_params({'n_estimators':n_estimators,'max_depth':max_depth})
        mlflow.sklearn.log_model(clf,clf.__class__.__name__)
        
def main(n_estimators: int ,max_depth: int):



     train_model(X_train=X_train_final,y_train=y_train,plot_name='imbalanced ',n_estimators=n_estimators,max_depth=max_depth,class_weight=dict_weight)
     
# Run via terminal
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--max_depth', '-m', type=int, default=80)
    
    args = parser.parse_args()
    
    # Call the main function
    main(n_estimators=args.n_estimators, max_depth=args.max_depth)
    

        
        
        
    
    
    



