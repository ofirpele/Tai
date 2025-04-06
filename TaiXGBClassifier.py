from TaiSpecialFeaturesTransformer import TaiSpecialFeaturesTransformer
from TaiLinearClassifier import TaiLinearClassifier

import xgboost as xgb
import numpy as np

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression

from dataclasses import dataclass
from typing import Any

@dataclass
class TaiXGBClassifier:

    with_constraints_from_logistic_regression : bool
    class_weight : Any
    logistic_regression_params_dict : dict = None

    def __post_init__(self):
        self.features_transformer = TaiSpecialFeaturesTransformer()
        if self.with_constraints_from_logistic_regression:
            self.linear_clf_for_monotone_constraints = TaiLinearClassifier(LogisticRegression(**self.logistic_regression_params_dict))

    def __str__(self):
        return f'{type(self).__name__} {self.with_constraints_from_logistic_regression=} {self.class_weight=}'
    
    def _fit_monotone_constraints(self, X_train, y_train):
        self.linear_clf_for_monotone_constraints.fit(X_train, y_train)
        coeff = self.linear_clf_for_monotone_constraints.linear_classifier.coef_[0]
        return tuple([
            int(np.sign(coeff[c] * self.linear_clf_for_monotone_constraints.scaler.scale_[c])) for c in range(len(coeff))
        ])

    def fit(self, X_train, y_train):
        self.features_transformer.fit(X_train)
        X_train = self.features_transformer.transform(X_train)
        
        if self.with_constraints_from_logistic_regression:
            self.clf = xgb.XGBClassifier(monotone_constraints=self._fit_monotone_constraints(X_train, y_train))
        else:            
            self.clf = xgb.XGBClassifier()
        
        sample_weights = compute_sample_weight(class_weight=self.class_weight, y=y_train)        
        self.clf.fit(X_train, y_train, sample_weight=sample_weights)

    def predict(self, X):
        X = self.features_transformer.transform(X)
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        X = self.features_transformer.transform(X)
        return self.clf.predict_proba(X)
