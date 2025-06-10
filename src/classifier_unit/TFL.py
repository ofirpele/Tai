#####################################################################################
import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_lattice as tfl
tf.keras.utils.disable_interactive_logging()

import logging
import sys
logging.disable(sys.maxsize)

# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras
#####################################################################################

import numpy as np
import pandas as pd
                                   
from dataclasses import dataclass
from dataclasses import InitVar

from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight

from ._internal.FitParams import FitParams

###############################################################################################
# used in catboost -> tfl
###############################################################################################
from .CatBoost import CatBoost
import json
# TODO_FUTURE make it more efficient
class _ListWithoutLesserOrEqualElements:

    def __init__(self):
        self.array = []

    def add(self, new_element):
        elements_to_keep = [False if x < new_element else True for x in self.array]
        should_add_new_element = True
        new_array = []
        for e, ek in zip(self.array, elements_to_keep):
            if ek:
                new_array.append(e)
                if new_element <= e:
                    should_add_new_element = False
        if should_add_new_element:
            new_array.append(new_element)
        self.array = new_array
###############################################################################################


@dataclass
class TFL:

    catboost_init_dict : dict

    learning_rate : float = 0.1
    batch_size : int = 128
    num_epochs : int = 5000
    verbose_predict : int = 0
    
    random_seed : InitVar[int] = 42

    catboost_model_tmp_filename : str = 'tmp_cb_model.json'

    def __post_init__(self, random_seed):
        tf.random.set_seed(random_seed)

    @staticmethod
    def convert_X_to_list_of_pd_series(X : np.ndarray):
        X = X.astype(float)
        return [pd.Series(X[:, f_i]) for f_i in range(X.shape[1])]

    def fit(self, p : FitParams):
        class_weight_vec = compute_class_weight(class_weight=p.class_weight, y=p.y_train, classes=np.unique(p.y_train))
        assert(len(class_weight_vec)==2)
        output_initialization = [class_weight_vec[0]*(-4/sum(class_weight_vec)), class_weight_vec[1]*(+4/sum(class_weight_vec))]
        del class_weight_vec

        cb_clf = CatBoost(**self.catboost_init_dict)
        cb_clf.fit(p)
        cb_clf.clf.save_model(self.catboost_model_tmp_filename, format="json")
        cb_clf_dict = json.load(open(self.catboost_model_tmp_filename, "r"))
        del cb_clf
        os.remove(self.catboost_model_tmp_filename)
        
        features_info_keys = list(cb_clf_dict['features_info'].keys())
        assert len(features_info_keys) == 1
        assert features_info_keys[0] == 'float_features'

        # TODO_FUTURE: allow getting the min and max from user (for when it is known but does not appear in training)
        features_min = np.min(p.X_train, axis=0) 
        features_max = np.max(p.X_train, axis=0)
        features_info = cb_clf_dict['features_info']['float_features']
        input_keypoints_of_feature = []
        for fi in range(len(features_info)):
            borders = features_info[fi]['borders']
            if borders is None:
                borders = []
            input_keypoints_of_feature.append(set(borders))
            input_keypoints_of_feature[fi].add(features_min[fi])
            input_keypoints_of_feature[fi].add(features_max[fi])
            input_keypoints_of_feature[fi] = sorted(input_keypoints_of_feature[fi])            
        
        num_features = p.X_train.shape[1]

        feature_appeared_in_a_lattice = [False] * num_features
        lattices = _ListWithoutLesserOrEqualElements()
        for curr_tree in cb_clf_dict['oblivious_trees']:
            lattice = set()
            curr_tree_depth = len(curr_tree["splits"])
            for depth in range(curr_tree_depth):
                f_i = curr_tree["splits"][depth]["float_feature_index"]
                feature_appeared_in_a_lattice[f_i] = True
                lattice.add(f'{f_i}')
            lattices.add(lattice)  
        new_lattices = []
        for lattice in lattices.array:
            new_lattices.append(sorted(lattice))
        lattices = new_lattices
        assert all(feature_appeared_in_a_lattice), "TODO_FUTURE: if not appeared in a lattice, either add it none the less with a separate lattice or directly or ignore it somehow"
        
        feature_configs = []
        for f_i in range(num_features):
            feature_configs.append(
                tfl.configs.FeatureConfig(
                    name=f'{f_i}',
                    pwl_calibration_input_keypoints=input_keypoints_of_feature[f_i],
                    monotonicity=p.monotone_constraints[f_i]
                )
            )

        config_params = {
           'output_initialization' : output_initialization,
           'interpolation' : 'simplex',
           'feature_configs' : feature_configs
        }

        if len(lattices) > 1:
            tfl_model_config = tfl.configs.CalibratedLatticeEnsembleConfig(
                lattices=lattices,
                separate_calibrators=True,
                **config_params
            )
            self.clf = tfl.premade.CalibratedLatticeEnsemble(tfl_model_config)
        else:
            tfl_model_config = tfl.configs.CalibratedLatticeConfig(
                **config_params
            )
            self.clf = tfl.premade.CalibratedLattice(tfl_model_config)

        self.clf.compile(
           loss=keras.losses.BinaryCrossentropy(from_logits=True),
           metrics=[keras.metrics.AUC(from_logits=True)],
           optimizer=keras.optimizers.Adam(self.learning_rate)
        )
        self.clf.fit(
          TFL.convert_X_to_list_of_pd_series(p.X_train),
          pd.Series(p.y_train),
          epochs=self.num_epochs,
          batch_size=self.batch_size,
          verbose=False,
          sample_weight = compute_sample_weight(p.class_weight, p.y_train)
        )        

    def predict_proba(self, X : np.ndarray):
        X_list_of_pd_series = TFL.convert_X_to_list_of_pd_series(X)
        tfl_y_predict_1_proba = tf.nn.sigmoid(self.clf.predict(X_list_of_pd_series, verbose=self.verbose_predict))
        tfl_predict_proba = np.empty((tfl_y_predict_1_proba.shape[0], 2))
        for i in range(tfl_y_predict_1_proba.shape[0]):
            tfl_predict_proba[i][1] = tfl_y_predict_1_proba[i][0]
            tfl_predict_proba[i][0] = 1.0 - tfl_y_predict_1_proba[i][0]
        return tfl_predict_proba
    
    ''' Custom pickle, saves the model config and weights. Does not save optimizer. For whole saving use model.clf.save
    '''
    def __getstate__(self):
        self_clf_json_config = self.clf.to_json()
        self_clf_weights = self.clf.get_weights()
                        
        state = self.__dict__.copy()
        state.pop('clf')
        state['self_clf_json_config'] = self_clf_json_config
        state['self_clf_weights'] = self_clf_weights
        
        return state
        
    def __setstate__(self, newstate):
         newstate['clf'] = keras.models.model_from_json(newstate['self_clf_json_config'], tfl.premade.get_custom_objects())
         newstate.pop('self_clf_json_config')
         newstate['clf'].set_weights( newstate['self_clf_weights'] )
         newstate.pop('self_clf_weights')
         self.__dict__.update(newstate)
