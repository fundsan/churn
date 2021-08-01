import os
import logging
#import churn_library_solution as cls
from churn_library import *

logging.basicConfig(
    filename='./logs/churn_script_logging_and_tests.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    test_df=pd.read_csv("./data/bank_data.csv")
    test_df=test_df[['Customer_Age','Dependent_count','Gender']]
    try:
        assert os.path.isdir("./images/eda")==True
        
    except AssertionError as err:
        logging.error("Testing eda: The directory to save file wasn't found")

    try:
        perform_eda(test_df)
        logging.info("Testing eda: directory to save analysis images  SUCCESSFULLY")
    except FileNotFoundError as err:
        logging.error("Testing eda: The directory to save file wasn't found")
        raise err
    try:
        assert os.path.isfile("./images/eda/Customer_Age_hist.png")==True
        os.remove("./images/eda/Customer_Age_hist.png") 
        assert os.path.isfile("./images/eda/Gender_dist.png")==True
        os.remove("./images/eda/Gender_dist.png") 
        assert os.path.isfile("./images/eda/corr.png")==True
        os.remove("./images/eda/corr.png") 
        logging.info("Testing eda: exploratory data analysis images saved SUCCESSFULLY")
        
    except AssertionError:
        logging.error("Testing eda: not all images are being saved")
        raise err
        
    

    

def test_encoder_helper():
    '''
    test encoder helper
    '''
    
    test_df=pd.read_csv("./data/test_bank_data.csv")
    test_df=test_df[['Attrition_Flag','Customer_Age','Dependent_count','Gender','Marital_Status']]
    categorical_test = ['Gender','Marital_Status']
    try :
        assert set(encoder_helper(test_df,categorical_test).columns)==set(['Attrition_Flag','Customer_Age','Dependent_count','Gender','Marital_Status','Churn','Gender_churn','Marital_Status_churn'])
        logging.info("Testing encoder_helper: SUCCESSFULLY Created new encoder columns")
        
    except AssertionError:
        logging.error("Testing encoder_helper: expected "+str(['Attrition_Flag','Customer_Age','Dependent_count','Gender','Marital_Status','Gender_churn','Marital_Status_churn'])+ " for columns but got "+str(encoder_helper(test_df,categorical_test).columns))
        raise err
        

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    test_df=pd.read_csv("./data/test_bank_data.csv")
   
    perform_feature_engineering(test_df, response='Churn')
    X_train, X_test, y_train, y_test =perform_feature_engineering(test_df, "Churn")
    try:
        assert set(X_train.columns) ==set(['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn'])
        logging.info("Testing perform_feature_engineering: SUCCESSFULLY Created Training and Test set with correct columns")
        except AssertionError:
            logging.error("Testing perform_feature_engineering: expected "+str(['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn'])+ " for columns but got "+str(X_train.columns))
        raise err
        

    
    

def test_train_models(train_models):
    '''
    test train_models
    '''
    test_df=pd.read_csv("./data/test_bank_data.csv")
    X_train, X_test, y_train, y_test =perform_feature_engineering(test_df, response='Churn')
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile('./models/rfc_model.pkl')==True
        os.remove("./images/eda/corr.png") 
        logging.info("Testing train_models: models SUCCESSFULLY saved")

"""
if __name__ == "__main__":
    test_import()
    test_eda()






