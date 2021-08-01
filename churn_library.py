# library doc string


# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import  classification_report

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    print('Null values for each variable:')
    print(df.isnull().sum())
    print('/n')
    print('Basic statistics of numerical data')
    print(df.describe())
    
    for column in list(df.describe().columns):
        ax = df[column].hist()
        fig = ax.get_figure()
        fig.savefig('images/eda/{}_hist.png'.format(column),bbox_inches='tight')
        plt.clf()
    category_columns = set(df.columns) - set(list(df.describe().columns))
    for column in list(category_columns):
        ax = df[column].value_counts('normalize').plot(kind='bar') 
        fig = ax.get_figure()
        fig.savefig('images/eda/{}_dist.png'.format(column),bbox_inches='tight')
        plt.clf()
    ax =sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig.savefig('images/eda/corr.png',bbox_inches='tight')
    plt.clf()
    


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for proportion of churn
    '''
    df[response] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    for category in category_lst:
        
        lst = []
        groups = df.groupby(category).mean()['Churn']

        
            

        df[category+'_'+response] = df[category].apply(lambda x:groups.loc[x] )
    return df

def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'  ]

    quant_columns = ['Customer_Age','Dependent_count', 'Months_on_book','Total_Relationship_Count', 'Months_Inactive_12_mon','Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal','Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt','Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    
    df = encoder_helper(df, cat_columns, response=response)
    
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X = df[keep_cols]
    y = df[response]
    # train test split 
    return train_test_split(X, y, test_size= 0.3, random_state=42)
    

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    fig=plt.figure()
    plt.rc('figure', figsize=(6, 6))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    fig.savefig('images/results/classification_report_{}.png'.format('rf'),bbox_inches='tight')
    plt.clf()
    fig=plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    fig.savefig('images/results/classification_report_{}.png'.format('lr'),bbox_inches='tight')
    plt.clf()
    
    


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    assert isinstance(model,RandomForestClassifier ) ==True
    # Calculate feature importances for rf
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(20,5))
    fig=plt.figure()
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    
    fig.savefig(output_pth,bbox_inches='tight')
    plt.clf()

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = { 'n_estimators': [200, 500]
        #,'max_features': ['auto', 'sqrt']
        #,'max_depth' : [4,5,100]
        #,'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    print("STARTING: Train Random Forest Grid Search")
    cv_rfc.fit(X_train, y_train)
    print("STARTING: Train Logistic Regression")
    lrc.fit(X_train, y_train)
    print("STARTING: Make Predictions")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    print("STARTING: Make Classification Images")
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    

if __name__ == "__main__":
    
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test =perform_feature_engineering(df, "Churn")
    train_models(X_train, X_test, y_train, y_test)
    print("SAVING Feature Importance Images")
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    feature_importance_plot(rfc_model, pd.concat([X_train,X_test]), 'images/results/rf_feature_importance.png')

    
    
    
    