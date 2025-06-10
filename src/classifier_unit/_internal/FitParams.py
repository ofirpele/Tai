from dataclasses import dataclass
import numpy as np

@dataclass
class FitParams:

    X_train : np.ndarray
    y_train : np.ndarray
    monotone_constraints : list[int]
    class_weight : np.ndarray
        