
# Python
import argparse
from helper import *
import mlflow
import os
import joblib

if __name__ =='__main__':

    # Start Logging
    mlflow.start_run()
    mlflow.sklearn.autolog() # enable autologging

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, help="Name of the Dataset")
    parser.add_argument("--model_name", type=str, help="Name of the Model")
    
    args = parser.parse_args()
    dataset    = args.dataset
    model_name = args.model_name
    
    # Processing Data
    X_train, X_val, y_train, y_val , unique_values_train , column_mapping = data_process(dataset, split_ratio=0.3)

    # Train Test Data
    # X_train, X_test, y_train, y_test = split_data(df, split_ratio=0.3)

    # Scale numeric columns and one-hot encode categorical columns
    X_train_scaled, X_val_scaled , scaler = scale_columns(X_train, X_val)

    # Training Model
    model = train_model(X_train_scaled, y_train)

    # Evaluate Model
    accuracy = model_metric(model, X_val_scaled , y_val)
    print('Accuracy:', accuracy)

    # Save model and other artifacts
    os.makedirs(model_name, exist_ok=True)

    scaler_path = f'{model_name}/scaler.pkl'
    categorical_path = f'{model_name}/unique_values_train.pkl'
    column_mapping_path = f'{model_name}/column_mapping.pkl'

    # dump    
    joblib.dump(value= scaler, filename=scaler_path)
    joblib.dump(value=unique_values_train, filename= categorical_path)
    joblib.dump(value= column_mapping, filename= column_mapping_path)



    # Registering Model
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model( sk_model=model,
                              registered_model_name=model_name,
                              artifact_path=model_name,
                            )

    # Stop Logging
    mlflow.end_run()

