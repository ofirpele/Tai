import xgboost as xgb

from sklearn.utils.class_weight import compute_sample_weight

from ._internal.FitParams import FitParams
from ._internal.InitBase import InitBase
from ._internal.PredictProbaBase import PredictProbaBase


class XGB(InitBase, PredictProbaBase):
    
    def fit(self, p : FitParams):
        self.clf = xgb.XGBClassifier(monotone_constraints=tuple(p.monotone_constraints), **self.kwargs_classifier_init)
        sample_weight = compute_sample_weight(class_weight=p.class_weight, y=p.y_train)
        self.clf.fit(p.X_train, p.y_train, sample_weight=sample_weight)