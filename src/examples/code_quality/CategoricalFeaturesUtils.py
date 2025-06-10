import numpy as np
from collections import defaultdict
import random

class CategoricalFeaturesUtils:

    def __init__(self, X_train, category_feature_column):
        self.X_train = X_train
        self.category_feature_column = category_feature_column

        self.map_category_to_one_hot_encoding_column = {}
        column = category_feature_column
        for val in X_train[:, self.category_feature_column]:
            if val not in self.map_category_to_one_hot_encoding_column:
                self.map_category_to_one_hot_encoding_column[val] = column
                column += 1
        self.num_categories = len(self.map_category_to_one_hot_encoding_column)
        self.one_hot_encoding_num_features = X_train.shape[1] - 1 + self.num_categories
    
    # col can be None if in X_train we did not see the category
    def _row_and_one_hot_encoding_category_col(self, X):
        for row in range(X.shape[0]):
            category = X[row, self.category_feature_column]
            col = self.map_category_to_one_hot_encoding_column.get(category, None)
            yield row, col

    def one_hot_encoding(self, X):
        X_out = np.zeros((X.shape[0], self.one_hot_encoding_num_features))
        for row, col in self._row_and_one_hot_encoding_category_col(X):
            assert col is not None
            assert self.category_feature_column == 0
            X_out[row, col] = 1
            X_out[row, self.num_categories:] = X[row, 1:]  
        return X_out
    
    def _row_and_row_out_for_categories(self, X, seen_categories = True):
        row_out = 0
        for row, col in self._row_and_one_hot_encoding_category_col(X):
            if (seen_categories and col is not None) or (not seen_categories and col is None):
                yield row, row_out
                row_out += 1

    def samples_with_seen_category(self, X, y):
        num_new_samples = sum(1 for _, _ in self._row_and_row_out_for_categories(X))
        
        X_out = np.empty((num_new_samples, X.shape[1]), dtype=X.dtype)
        y_out = np.empty((num_new_samples,), dtype=y.dtype)
        for row, row_out in self._row_and_row_out_for_categories(X):
            X_out[row_out, :] = X[row, :]
            y_out[row_out] = y[row]

        return X_out, y_out
    
    def _row_and_row_out_one_in_category(self, X, location):
        category_to_rows = defaultdict(list)
        for row in range(X.shape[0]):
            category_to_rows[X[row, self.category_feature_column]].append( row )
        row_out = 0
        for category, rows in category_to_rows.items():
            match location:
                case 'first':
                    ind = 0
                case 'middle':
                    ind = (len(rows) - 1)//2
                case 'last':
                    ind = -1
                case 'random':
                    ind = random.randint(0, len(rows)-1)
                case _:
                    assert False, f'{location=} is not supported'
            yield rows[ind], row_out
            row_out += 1            
        
    # location can be: 'first', 'middle', 'last', 'random'
    def samples_one_in_category(self, X, y, location):
        num_new_samples = sum(1 for _, _ in self._row_and_row_out_one_in_category(X, location))
        X_out = np.empty((num_new_samples, X.shape[1]), dtype=X.dtype)
        y_out = np.empty((num_new_samples,), dtype=y.dtype)
        for row, row_out in self._row_and_row_out_one_in_category(X, location):
            X_out[row_out, :] = X[row, :]
            y_out[row_out] = y[row]
        return X_out, y_out
        
        
    def samples_with_categories_not_seen_in_train(self, X, y):
        num_new_samples = sum(1 for _, _ in self._row_and_row_out_for_categories(X, seen_categories = False))
        X_out = np.empty((num_new_samples, X.shape[1]), dtype=X.dtype)
        y_out = np.empty((num_new_samples,), dtype=y.dtype)
        for row, row_out in self._row_and_row_out_for_categories(X, seen_categories = False):
            X_out[row_out, :] = X[row, :]
            y_out[row_out] = y[row]
        return X_out, y_out
    


def test_init():
    X_train = np.array(
        [['A', '1', 3],
         ['B', '10', 5],
         ['A', '3', 6],
         ['C', '3', 55]])
    
    cfu = CategoricalFeatureUtils(X_train, 0)
    assert cfu.map_category_to_one_hot_encoding_column == {'A':0, 'B':1, 'C':2}

    cfu = CategoricalFeatureUtils(X_train, 1)
    assert cfu.map_category_to_one_hot_encoding_column == {'1':1, '10':2, '3':3}

def test_one_hot_encoding():
    X_train = np.array([
        [11, 1, 3],
        [22, 10, 5],
        [11, 3, 6]
    ])
    cfu = CategoricalFeatureUtils(X_train, 0)
    
    X_train_out = cfu.one_hot_encoding(X_train)
    assert (X_train_out == np.array([
        [1, 0, 1, 3],
        [0, 1, 10, 5],
        [1, 0, 3, 6]
    ])).all()

    X_test = np.array([
        [11, 1.5, 3.5],
        [22, 10, 5],
    ])
    X_test_out = cfu.one_hot_encoding(X_test)
    assert (X_test_out == np.array([
        [1, 0, 1.5, 3.5],
        [0, 1, 10, 5]        
    ])).all()

def test_samples_with_seen_category_in_train_1():
    X_train = np.array([
        [11, 1, 3],
        [22, 10, 5],
        [11, 3, 6]
    ])
    cfu = CategoricalFeatureUtils(X_train, 0)

    X_test = np.array([
        [11, 1.5, 3.5],
        [22, 10, 5],
        [33, 3, 6]
    ])
    y_test = np.array([
        0, 
        -11, 
        33
    ])
    X_test_out, y_test_out = cfu.samples_with_seen_category(X_test, y_test)
    assert (X_test_out == np.array([
        [11, 1.5, 3.5],
        [22, 10, 5]
    ])).all()
    assert (y_test_out == np.array([
        0, 
        -11
    ])).all() 

def test_samples_with_seen_category_in_train_1():
    X_train = np.array([
        [33, 1, 3],
        [22, 10, 5],
        [33, 3, 6]
    ])
    cfu = CategoricalFeatureUtils(X_train, 0)

    X_test = np.array([
        [11, 1.5, 3.5],
        [22, 10, 5],
        [33, 3, 6]
    ])
    y_test = np.array([
        0, 
        -11, 
        33
    ])
    X_test_out, y_test_out = cfu.samples_with_seen_category(X_test, y_test)
    assert (X_test_out == np.array([
        [22, 10, 5],
        [33, 3, 6]
    ])).all()
    assert (y_test_out == np.array([
        -11, 
        33
    ])).all() 

def test_samples_only_one_in_category():
    X_train = np.array([
        [11, 1, 3],
        [11, 3, 6],
        [22, 10, 5],
        [11, 33, 63],
        [22, 110, 15],
        [22, 1111, -11]    
    ])
    y_train = np.array([
        111,
        222,
        333,
        444,
        555,
        666
    ])

    X_train_out_expected = np.array([
        [11, 1, 3],
        [22, 10, 5]
    ])
    y_train_out_expected = np.array([
        111,
        333
    ])
    cfu = CategoricalFeatureUtils(X_train, 0)
    X_train_out, y_train_out = cfu.samples_one_in_category(X_train, y_train, 'first')# .samples_only_first_in_category(X_train, y_train)
    assert (X_train_out == X_train_out_expected).all()
    assert (y_train_out == y_train_out_expected).all()

    X_train_out_expected = np.array([
        [11, 3, 6],
        [22, 110, 15]
    ])
    y_train_out_expected = np.array([
        222,
        555
    ])
    cfu = CategoricalFeatureUtils(X_train, 0)
    X_train_out, y_train_out = cfu.samples_one_in_category(X_train, y_train, 'middle')# .samples_only_first_in_category(X_train, y_train)
    assert (X_train_out == X_train_out_expected).all()
    assert (y_train_out == y_train_out_expected).all()


def test_samples_with_categories_not_seen_in_train():
    X_train = np.array([
        [11, 1, 3],
        [11, 3, 6],
        [22, 10, 5],
        [11, 33, 63],
        [22, 110, 15]
    ])
    X_test = np.array([
        [11, 1, 3],
        [3, 3, 6],
        [22, 10, 5],
        [11, 33, 63],
        [4, 110, 15],
        [22, 33, 63]
    ])
    X_test_out_expected = np.array([
        [3, 3, 6],
        [4, 110, 15]
    ])
    y_test = np.array([
        111,
        222,
        333,
        444,
        555,
        666
    ])
    y_test_out_expected = np.array([
        222,
        555
    ])
    cfu = CategoricalFeatureUtils(X_train, 0)
    X_test_out, y_test_out = cfu.samples_with_categories_not_seen_in_train(X_test, y_test)
    assert (X_test_out == X_test_out_expected).all()
    assert (y_test_out == y_test_out_expected).all()

