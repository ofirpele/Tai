from typing import Protocol
from .FitParams import FitParams

import numpy as np

class Protocol:

    def fit(self, p : FitParams):
        ...

    def predict_proba(self, X : np.ndarray):
        ...
  