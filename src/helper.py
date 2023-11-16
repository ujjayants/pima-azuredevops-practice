
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# from imblearn.over_sampling import SMOTE
import os
import pandas as pd
#import mlflow
#import logging
#import joblib


def data_process(dataset, split_ratio): ## Add split ratio

    # Loading processed data   
    df = pd.read_csv(dataset)

    column_mapping = { df.columns[0]:'Pregnancies',df.columns[1] :'Glucose',
    df.columns[2]:'BloodPressure', df.columns[3] : 'SkinThickness',
    df.columns[4]: 'Insulin' , df.columns[5]:  'BMI',
    df.columns[6] : 'DiabetesPedigreeFunction' , df.columns[7] : 'Age' ,
    df.columns[8] :'Outcome'}

    # Rename columns using the rename() function
    df.rename(columns=column_mapping, inplace=True)

    X_train, X_val, y_train, y_val = split_data(df, split_ratio)

    X_train, X_val = impute_mv(X_train, X_val)

    # BMI
    X_train = X_train.assign(BM_DESC= X_train.apply(set_bmi, axis=1))
    X_val = X_val.assign(BM_DESC= X_val.apply(set_bmi, axis=1))

    # Change to type categorical
    X_train['BM_DESC'] = X_train['BM_DESC'].astype('category')
    X_val['BM_DESC'] = X_val['BM_DESC'].astype('category')


    # Insulin
    X_train = X_train.assign(INSULIN_DESC= X_train.apply(set_insulin, axis=1))
    X_val = X_val.assign(INSULIN_DESC= X_val.apply(set_insulin, axis=1))

    # Change to type categorical
    X_train['INSULIN_DESC'] = X_train['INSULIN_DESC'].astype('category')
    X_val['INSULIN_DESC'] = X_val['INSULIN_DESC'].astype('category')

    # Identify unique categorical values in the training set
    unique_values_train =  {col: X_train[col].unique() for col in X_train.select_dtypes(include=['category']).columns}

    for col in X_val.columns:
        if col in unique_values_train:
            # Ensure that the validation set's categorical column has the same unique values as the training set
            X_val[col] = X_val[col].astype('category').cat.set_categories(unique_values_train[col])

    return  X_train, X_val, y_train, y_val , unique_values_train , column_mapping  #df


# Returns one-hot encoded dataframes and numeric columns
def one_hot_encoding(df_X_train, df_X_val):

    # One-hot encoding
    cols_drop=['Insulin','BMI']
    df_X_train= df_X_train.drop(cols_drop,axis=1)
    df_X_val= df_X_val.drop(cols_drop,axis=1)

    # Select only numeric columns
    numeric_columns = df_X_train.select_dtypes(include=['number']).columns.tolist()

    df_X_train= pd.get_dummies(df_X_train)
    df_X_val= pd.get_dummies(df_X_val)

    return df_X_train, df_X_val, numeric_columns


# Apply min-max scaling to numeric columns
def scale_columns(df_X_train, df_X_val):

    # One-hot encoding
    X_train, X_val, numeric_columns = one_hot_encoding(df_X_train, df_X_val)

    # Initialize the StandardScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the data (computes the minimum and maximum values)
    scaler.fit(X_train[numeric_columns])

    # Train data- Transform the data using the computed minimum and maximum values
    X_train_num_scaled = scaler.transform(X_train[numeric_columns])

    # Concatenate scaled numeric columns and categorical columns
    X_train_scaled = pd.concat([  pd.DataFrame(X_train_num_scaled, columns= numeric_columns).reset_index(drop= True)
    ,  X_train[ X_train.columns.difference(numeric_columns)  ].reset_index(drop= True)  ] , axis =1  )

    # Val data- Transform the data using the computed minimum and maximum values
    X_val_num_scaled = scaler.transform(X_val[numeric_columns])

    # Concatenate scaled numeric columns and categorical columns
    X_val_scaled = pd.concat([pd.DataFrame(X_val_num_scaled, columns= numeric_columns).reset_index(drop= True)  ,
            X_val[ X_val.columns.difference(numeric_columns)  ].reset_index(drop= True) ], axis = 1)

    del X_train, X_val, X_train_num_scaled, X_val_num_scaled

    return X_train_scaled, X_val_scaled, scaler


# Split data into train and validation
def split_data(df, split_ratio):

    # Split Data
    X = df.loc[:, df.columns != 'Outcome']
    y =  df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = split_ratio, 
                                                        random_state = 42,  
                                                        stratify = y
                                                       )
    
    return X_train, X_test, y_train, y_test


# Missing data imputation using median
def impute_mv(df_train, df_val):

    # Calculate the median value for BMI
    median_bmi = df_train['BMI'].median()
    # Substitute it in the BMI column of the
    # dataset where values are 0
    df_train['BMI'] = df_train['BMI'].replace(
        to_replace=0, value=median_bmi)
    
    df_val['BMI'] = df_val['BMI'].replace(
        to_replace=0, value=median_bmi)

    median_bloodp = df_train['BloodPressure'].median()
    # Substitute it in the BloodP column of the
    # dataset where values are 0
    df_train['BloodPressure'] = df_train['BloodPressure'].replace(
        to_replace=0, value=median_bloodp)
    
    df_val['BloodPressure'] = df_val['BloodPressure'].replace(
        to_replace=0, value=median_bloodp)

    # Calculate the median value for PlGlcConc
    median_plglcconc = df_train['Glucose'].median()
    # Substitute it in the PlGlcConc column of the
    # dataset where values are 0
    df_train['Glucose'] = df_train['Glucose'].replace(
        to_replace=0, value=median_plglcconc)

    df_val['Glucose'] = df_val['Glucose'].replace(
        to_replace=0, value=median_plglcconc)


    # Calculate the median value for SkinThick
    median_skinthick = df_train['SkinThickness'].median()
    # Substitute it in the SkinThick column of the
    # dataset where values are 0
    df_train['SkinThickness'] = df_train['SkinThickness'].replace(
        to_replace=0, value=median_skinthick)

    df_val['SkinThickness'] = df_val['SkinThickness'].replace(
        to_replace=0, value=median_skinthick)


    # Calculate the median value for SkinThick
    median_skinthick = df_train['Insulin'].median()
    # Substitute it in the SkinThick column of the
    # dataset where values are 0
    df_train['Insulin'] = df_train['Insulin'].replace(
        to_replace=0, value=median_skinthick)
    
    df_val['Insulin'] = df_val['Insulin'].replace(
        to_replace=0, value=median_skinthick)
    
    return df_train, df_val


# Convert BMI values to categories
def set_bmi(row):
    if row["BMI"] < 18.5:
        return "Under"
    elif row["BMI"] >= 18.5 and row["BMI"] <= 24.9:
        return "Healthy"
    elif row["BMI"] >= 25 and row["BMI"] <= 29.9:
        return "Over"
    elif row["BMI"] >= 30:
        return "Obese"


# Convert Insulin values to categories
def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"




def train_model(X_train, y_train):
    
    # Training Model
    model = RandomForestClassifier(class_weight='balanced',
                                    bootstrap=True,
                                    max_depth=100,
                                    max_features=2,
                                    min_samples_leaf=5,
                                    min_samples_split=10,
                                    n_estimators=1000,
                                    random_state = 42
                                    )
    # without Sampling
    model.fit(X_train, y_train)

    return model

def model_metric(model, X_test, y_test):

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    return accuracy
