import numpy as np

class PredictProbaBase:
    
    def predict_proba(self, X : np.ndarray):
        return self.clf.predict_proba(X)