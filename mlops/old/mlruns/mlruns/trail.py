""""
import mlflow
import argparse



exp= mlflow.set_experiment('test')
with mlflow.start_run(run_name='kenzy') as run:
    mlflow.log_param("param1", 10)
    mlflow.log_param("param2", 20)
    
    mlflow.log_params({' param1' : 10,' param2 ': 20}) 
    
def eval(param1 : int, param2: int):
    return(param1 + param2)/2


def main(param1 : int, param2: int):
     mlflow.set_experiment(experiment_name='Kenzy_test')
     with mlflow.start_run() as run:
         
         #logging for params
         mlflow.log_param("param1",param1)
         mlflow.log_param("param2",param2)
         
         #logging metric
         mlflow.log_metric("average", eval(param1=param1,param2=param2))
         
#Run vi terminal 
if __name__ =='__main__':
    
    #parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--param1', '-p1',type=int ,default=10)
    parser.add_argument('--param2', '-p2',type=int ,default=20)
    
    
    args = parser.parse_args()   
    
    args.param1 , args.param2
    
    #call the main function
    main(param1=args.param1, param2=args.param2)
    
"""
import mlflow
import argparse

def eval(param1: int, param2: int):
    return (param1 + param2) / 2

def main(param1: int, param2: int):
    mlflow.set_experiment(experiment_name='Kenzy_test')
    with mlflow.start_run() as run:
        # Logging parameters
        mlflow.log_param("param1", param1)
        mlflow.log_param("param2", param2)
        
        # Logging metric
        mlflow.log_metric("average", eval(param1=param1, param2=param2))

# Run via terminal
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--param1', '-p1', type=int, default=10)
    parser.add_argument('--param2', '-p2', type=int, default=20)
    
    args = parser.parse_args()
    
    # Call the main function
    main(param1=args.param1, param2=args.param2)
