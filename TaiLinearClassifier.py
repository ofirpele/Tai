from TaiSpecialFeaturesTransformer import TaiSpecialFeaturesTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import numpy as np

class TaiLinearClassifier:

    def __init__(self, linear_classifier):
        self.taiSpecialFeaturesTransformer = TaiSpecialFeaturesTransformer()
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = MinMaxScaler()
        self.linear_classifier = linear_classifier
        self.clf = Pipeline([
            ('taiSpecialFeaturesTransformer', self.taiSpecialFeaturesTransformer),
            ('imputer', self.imputer),
            ('scaler', self.scaler),
            ('classifier', linear_classifier)
        ])
        
    def __str__(self):
        return f'{type(self).__name__} where linear_classifier={self.linear_classifier}'
    
    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

   