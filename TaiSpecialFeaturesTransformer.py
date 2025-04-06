import numpy as np

# Removes columns with single value (including all nans)
# For columns that have nan and a single non-nan value:
# - The nan is mapped to 0 
# - The non-nan is mapped to 1
class TaiSpecialFeaturesTransformer:

    def fit(self, X_train, _=None):
        self.cols_to_remove = []
        self.cols_one_nan_one_not_nan = []
        for col_num in range(X_train.shape[1]):
            
            is_there_nan = False
            is_there_at_least_one_value_not_nan = False
            first_value_not_nan = None
            is_there_at_least_two_values_not_nan = False
            for row_num in range(X_train.shape[0]):
                val = X_train[row_num, col_num]
                if np.isnan(val):
                    is_there_nan = True
                elif not is_there_at_least_one_value_not_nan:
                    is_there_at_least_one_value_not_nan = True
                    first_value_not_nan = val
                elif val != first_value_not_nan:
                    is_there_at_least_two_values_not_nan = True
                    break
            
            if is_there_nan:
                if not is_there_at_least_one_value_not_nan:
                    # only nans
                    self.cols_to_remove.append(col_num)
                elif not is_there_at_least_two_values_not_nan:
                    self.cols_one_nan_one_not_nan.append(col_num)                    
            elif not is_there_at_least_two_values_not_nan:
                # only one value
                self.cols_to_remove.append(col_num)
        return self                

    def transform(self, X):
        for col in self.cols_one_nan_one_not_nan:
            for row in range(X.shape[0]):
                if np.isnan(X[row, col]):
                    X[row, col] = 0
                else:
                    X[row, col] = 1
        return np.delete(X, self.cols_to_remove, axis=1)


def test_1():
    # 0 all nan - remove
    # 1 >=2 values not nan: ok
    # 2 two values one nan: change nan to 0 and other value to 1
    # 3 >=2 values not nan: ok
    # 4 >=2 values not nan: ok
    # 5 >=2 values not nan: ok
    # 6 single value:       remove
    # 7 >=2 values not nan: ok
    X_train = np.array([        
        [np.nan, 1,         2,      3, 4, 5,      6, 7],
        [np.nan, np.nan,    2,      4, 5, np.inf, 6, 71],
        [np.nan, 3,         np.nan, 3, 6, 5,      6, 7],
    ])

    t = TaiSpecialFeaturesTransformer()
    t.fit(X_train)
    X_train_out = t.transform(X_train)

    assert np.isnan(X_train_out[1, 0])
    X_train_out[1,0] = 666
    assert (X_train_out == np.array([
        [1,         1,      3, 4, 5,      7],
        [666,       1,      4, 5, np.inf, 71],
        [3,         0,      3, 6, 5,      7],
        
    ])).all()
