from config import config
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


""" def extract_only_letter (_df):
    df['cabin'] = df['cabin'].str[0]
    df=df.reset_index()
    return df
 """

""" def numerical_encoding (_df ,NUMERICAL_FEATURES):

    for var in NUMERICAL_FEATURES:

        

        # replace NaN by median
        median_val = _df[var].median()

        _df[var].fillna(median_val, inplace=True)

    return _df  """



class NumericalImputer(BaseEstimator,TransformerMixin):
    """Numerical Data Missing Value Imputer"""
    def __init__(self, variables=None):
            self.variables = variables
    
    def fit(self, X,y=None):
        self.imputer_dict_={}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].median()
        return self

    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature],inplace=True)
        return X
