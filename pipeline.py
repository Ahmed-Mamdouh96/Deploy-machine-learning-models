from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import processing.preprocessor as pp 
from sklearn.linear_model import LogisticRegression

from config import config

pipeline = Pipeline(
[
    ("Numerical_Impute",pp.NumericalImputer(variables=config.NUMERICAL_FEATURES)),

    ("Standard", StandardScaler()),

    ("Model",  LogisticRegression(C=0.5, random_state=0))

]



)