"""
#dataset
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 
"""
#import the libraries 


#import the libraries 
import mlflow
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Read the dataset
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(data_url, sep=';')

# Split the dataset 
X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1

def main(n_estimators: int, max_depth: int):
    mlflow.set_experiment('basicml')
    
    with mlflow.start_run() as run:
        
        # Logging Tag
        mlflow.set_tag('clf', 'forest')
        
        # Logging params
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        
        # Train the model
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=45)
        clf.fit(X_train, y_train)
        
        y_pred_test = clf.predict(X_test)
        
        # Evaluate 
        accuracy, f1 = evaluate(y_test, y_pred_test)
        
        # Logging metrics 
        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('F1', f1)
        
        # Logging the model
        mlflow.sklearn.log_model(clf, "random_forest_model")

# Run via terminal
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--max_depth', '-m', type=int, default=80)
    
    args = parser.parse_args()
    
    # Call the main function
    main(n_estimators=args.n_estimators, max_depth=args.max_depth)
