import os
import logging
import json
import numpy as np
import mlflow
import joblib
import pandas as pd



# Called when the service is loaded
def init():

    try:
        global model, scaler, unique_train_values, column_mapping

        # Method 1
        # model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'credit_defaults_model_SDKv2','model.pkl')
        # model = joblib.load(model_path)

        # Method 2: unpickle using mlflow
        # Note: provide folder name as is in artifact
        model_name = 'pimaModelCI'
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name)
        print( 'Model path- ', model_path)
        model = mlflow.sklearn.load_model(model_path)

        scaler = joblib.load( os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name ,'scaler.pkl')  )
        unique_train_values = joblib.load(os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name ,'unique_values_train.pkl'))
        column_mapping = joblib.load(os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name  ,'column_mapping.pkl'))


        # Method 2
        # model_name = 'credit_defaults_model_SDKv2'
        # # to get latest version
        # stage = 'latest'
        # model_uri = f"models:/{model_name}/{stage}"  
        # model = mlflow.sklearn.load_model(model_uri=model_uri)  
        
        logging.info("Init complete")
        
    except Exception as e:
        print('Exception occured:', e)
    

## Function to convert BMI into categories
def set_bmi(row):
    if row["BMI"] < 18.5:
        return "Under"
    elif row["BMI"] >= 18.5 and row["BMI"] <= 24.9:
        return "Healthy"
    elif row["BMI"] >= 25 and row["BMI"] <= 29.9:
        return "Over"
    elif row["BMI"] >= 30:
        return "Obese"


## Function to convert Insulin into categories
def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


## Function to prepare data for prediction
def data_pred_prep(json_input):

    df_json  =  pd.DataFrame(json_input, 
    columns = [value for key,value in column_mapping.items() if value != 'Outcome'])

    # BMI
    df_json = df_json.assign(BM_DESC= df_json.apply(set_bmi, axis=1))

    # Change to type categorical
    df_json['BM_DESC'] = df_json['BM_DESC'].astype('category')

    # Insulin
    df_json = df_json.assign(INSULIN_DESC= df_json.apply(set_insulin, axis=1))

    # Change to type categorical
    df_json['INSULIN_DESC'] = df_json['INSULIN_DESC'].astype('category')

    for col in df_json.columns:
        if col in unique_train_values:
            # Ensure that the validation set's categorical column has the same unique values as the training set
            df_json[col] = df_json[col].astype('category').cat.set_categories(unique_train_values[col])


    # One-hot encoding
    cols_drop=['Insulin','BMI']
    df_json = df_json.drop(cols_drop,axis=1)

    # Select only numeric columns
    numeric_columns = df_json.select_dtypes(include=['number']).columns.tolist()

    df_json = pd.get_dummies(df_json)

    # Transform the data using the computed minimum and maximum values
    df_json_num_scaled = scaler.transform(df_json[numeric_columns])

    df_json_scaled = pd.concat([  pd.DataFrame(df_json_num_scaled, columns= numeric_columns).reset_index(drop= True)
    ,  df_json[ df_json.columns.difference(numeric_columns)  ].reset_index(drop= True)  ] , axis =1  )


    del df_json, df_json_num_scaled

    return df_json_scaled




def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    try:
        logging.info("Request received")
        data = json.loads(raw_data)["data"]
        data = np.array(data)
        result = model.predict(data_pred_prep(data))
        logging.info("Request processed")
        return result.tolist()
    except Exception as e:
        print('Exception Occured in run():', e)