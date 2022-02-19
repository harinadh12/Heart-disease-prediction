import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, make_scorer, classification_report,plot_roc_curve
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFE,SelectKBest, f_classif,chi2,mutual_info_classif, SelectFromModel
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesClassifier






class HeartDiseasePrediction():
    def __init__(self, data):
        """ 
        Constructor for initializing data 
        """
        self.data = data
        self.X =  None
        self.y = None
       
                
    def prepare_data(self):
        """ 
        This method prepares data with cateforical features and numerical features 
        """

        self.data.replace('?',np.nan, inplace = True)
        data = self.data
        data.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','heart_failure']
        data['heart_failure'] =np.where(data['heart_failure'] >0, 1, 0)
        cat_cols = ['sex','cp','fbs','restecg','exang','slope','thal','heart_failure']
        data[cat_cols] = data[cat_cols].astype('category')
        num_cols = list(set(data.columns)-set(cat_cols))
        data[num_cols] = data [num_cols].astype('float64')
        self.data = data
        
    def filter_missing_data(self):
        """ 
        Removes missing rows from the data
        """
        self.data.dropna(axis= 0, inplace=True)
        
    def data_split(self):
        """ 
        Splits data into train and test folds
        """
        self.data = pd.get_dummies(self.data, columns = self.data.select_dtypes('category').columns[:-1])
        self.X, self.y = self.data.loc[:,self.data.columns!='heart_failure'], self.data.loc[:,self.data.columns == 'heart_failure']        

    def standardize_data(self):
        """ 
        Data Standardization of only numeric features  with mean of 0 and standard deviation of 0    
        """
        standard_scaler = preprocessing.StandardScaler()
        self.X[self.X.columns] = standard_scaler.fit_transform(self.X)
   
    def train_model(self, model_name = 'SVM', n_folds = 10):
        """
        Model Training using n_folds cross validation and given model_name 
        Grid Search is also performed for svm to select best parameters
        """

        cv = KFold(n_splits = n_folds, random_state = 0, shuffle = True)
        if model_name == 'LR':
            print("******************** Logistic Regression *****************")
            self.model = LogisticRegression(random_state = 0)
            
        elif model_name == 'SVM':
            print("******************** Support Vector Machines *****************")
            svm_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear', 'poly','sigmoid']}
            grid = GridSearchCV(SVC(random_state=0), svm_grid, verbose = 0, cv = cv)
            grid.fit(self.X, self.y)
            best_params = grid.best_params_
            self.model = SVC(C = best_params.get('C'), gamma=best_params.get('gamma'), kernel=best_params.get('kernel'), random_state=0)


        specificity = make_scorer(recall_score, pos_label = 0)
        sensitivity = make_scorer(recall_score, pos_label = 1)
        accuracy = make_scorer(accuracy_score)
        param_grid = {'sensitivity': sensitivity,
                    'specificity': specificity,
                    'accuracy':'accuracy',
                    'roc_auc':'roc_auc'}

        cv_results = cross_validate(self.model, self.X, self.y, scoring = param_grid, cv = cv)

        print("########### Mean CV Accuracy ######## -- ",np.mean(cv_results.get('test_accuracy')))
        print("###### Mean CV Specificity ####### -- ",np.mean(cv_results.get('test_specificity')) ) 
        print("###### Mean CV Sensitivity ####### -- ",np.mean(cv_results.get('test_sensitivity')))
        print("########## Mean CV G-Mean ####### -- ", np.mean(np.sqrt(cv_results.get('test_specificity')*cv_results.get('test_sensitivity'))))
        print('######### Mean CV AUC of ROC curve ######## -- ', np.mean(cv_results.get('test_roc_auc')))
        
    def feature_selection(self):
        """ 
        Peform feature selection before fitting model on the training data
        i) Used model based sequential feature selection method to select best 10 features out of 25
        """

        fit = SelectFromModel(SVC(kernel='linear', random_state=0)).fit(self.X, self.y)
        self.X = self.X.iloc[:,fit.get_support(indices=True)]
        self.train_model(model_name='SVM')
        self.train_model(model_name='LR')

if __name__ == "__main__":

    data = pd.read_csv('processed.cleveland.data',header=None)
    h1 = HeartDiseasePrediction(data)
    h1.prepare_data()
    h1.filter_missing_data()
    h1.data_split()
    h1.standardize_data()
    lr  = h1.train_model(model_name='LR')

    svm = h1.train_model(model_name= 'SVM')
    h1.feature_selection()
