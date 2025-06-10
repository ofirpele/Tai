from sklearn.linear_model import LogisticRegression as LR

from sklearn.utils.class_weight import compute_sample_weight

from ._internal.FitParams import FitParams
from ._internal.InitBase import InitBase
from ._internal.PredictProbaBase import PredictProbaBase

# TODO_FUTURE
import warnings


class LogisticRegression(InitBase, PredictProbaBase):
    
    def fit(self, p : FitParams):
        self.clf = LR(**self.kwargs_classifier_init)
        warnings.warn('TODO_FUTURE: check that monotone constraints are consistent, currently does not check it')
        sample_weight = compute_sample_weight(class_weight=p.class_weight, y=p.y_train)
        self.clf.fit(p.X_train, p.y_train, sample_weight=sample_weight)
   