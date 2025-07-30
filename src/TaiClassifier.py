import numpy as np

from sklearn.linear_model import LogisticRegression
                                    
from dataclasses import dataclass
from typing import Any

from TaiSpecialFeaturesTransformer import TaiSpecialFeaturesTransformer
from TaiLinearClassifier import TaiLinearClassifier

from classifier_unit._internal.FitParams import FitParams
from classifier_unit._internal.Protocol import Protocol
# for combining
from classifier_unit.TFL import TFL 

@dataclass
class TaiClassifier:
    
    classifiers : list[Protocol]
     
    with_constraints_from_logistic_regression : bool = False
    logistic_regression_params_dict : dict[str, Any] = None
    
    class_weight : Any = 'balanced'
    
    features_names : list[str] = None

    def __post_init__(self):
        assert self.with_constraints_from_logistic_regression or self.logistic_regression_params_dict is None
        
        self.features_transformer = TaiSpecialFeaturesTransformer()
        if self.with_constraints_from_logistic_regression:
            self.linear_clf_for_monotone_constraints = TaiLinearClassifier(LogisticRegression(**self.logistic_regression_params_dict))

        if len(self.classifiers) > 1:
            # TODO_FUTURE: get catboost_init_dict from user? get random seed from user?
            self.combined_clf = TFL(catboost_init_dict={'random_seed' : 42, 'allow_writing_files' : False, 'silent' : True})
            
    def __str__(self):
        res = ''
        if self.with_constraints_from_logistic_regression:
            res += f'{type(self).__name__}: '
        elif len(self.classifiers) > 1:
            res += 'Ensemble: '
        res += '['
        for clf in self.classifiers:
            res += f'{type(clf).__name__} '
        res = res[:-1] # remove last space
        res += ']'
        res += f' class_weight={self.class_weight}'
        if not self.with_constraints_from_logistic_regression:
            res+= ' No Monotone'
        return res
   
    def _fit_monotone_constraints(self, X_train, y_train):
        self.linear_clf_for_monotone_constraints.fit(X_train, y_train)
        coeff = self.linear_clf_for_monotone_constraints.linear_classifier.coef_[0]    
        # res = [0]*len(coeff) 
        # for i, c in enumerate(coeff):
        #     if not (-1.0 <= c <= 1.0):
        #         res[i] = int(np.sign(c))
        # print([f'{self.active_features_names[c]}={coeff[c]:.1f}={res[c]}' for c in range(len(coeff))])
        return [int(np.sign(coeff[c])) for c in range(len(coeff))]
           
    def _X_of_classifiers_proba_1(self, X):
        X_of_proba_1 = np.empty((X.shape[0], len(self.classifiers)))
        for clf_i, clf in enumerate(self.classifiers):
            proba = clf.predict_proba(X)
            X_of_proba_1[:, clf_i] = proba[:, 1]
        return X_of_proba_1
            
    def fit(self, X_train, y_train):
        if self.features_names is None:
            self.features_names = [f'x{i}' for i in range(X_train.shape[1])]

        self.features_transformer.fit(X_train)
        X_train = self.features_transformer.transform(X_train)
        self.active_features_names = [self.features_names[c] for c in self.features_transformer.cols_to_include()]
        
        if self.with_constraints_from_logistic_regression:
            self.monotone_constraints = self._fit_monotone_constraints(X_train, y_train)
        else:
            self.monotone_constraints = [0] * len(self.active_features_names) 
        
        p = FitParams(X_train, y_train, self.monotone_constraints, self.class_weight)
        for clf in self.classifiers:
            clf.fit(p)

        if len(self.classifiers)>1:
            p = FitParams(self._X_of_classifiers_proba_1(X_train), y_train, [+1]*len(self.classifiers), self.class_weight)
            self.combined_clf.fit(p)
                    
    def predict_proba(self, X):
        X = self.features_transformer.transform(X)
        if len(self.classifiers)>1:
            return self.combined_clf.predict_proba(self._X_of_classifiers_proba_1(X))
        else:
            return self.classifiers[0].predict_proba(X)
