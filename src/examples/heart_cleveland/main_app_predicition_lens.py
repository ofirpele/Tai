import os, pickle
import prediction_lens
from pathlib import Path

MODEL_FILENAME = 'TaiClassifier_TFL.pkl'

MODELS_DIR_PATH = (
    os.path.join(Path(__file__).resolve().parents[0], "models") + "\\"
)
filename = MODELS_DIR_PATH + MODEL_FILENAME
with open(filename, 'rb') as fp:
    clf = pickle.load(fp)

features_init_value = [ 29.,   0.,  94., 126.,   0.,  71.,   0.,   0.,   0.,   0.,   0., 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]
features_is_only_min_and_max = [False, True, False, False, True, False, True, False, False, False, True, True, True, True, True, True, True, True, True, True, ]
prediction_lens.make_and_run_app(
    clf, 
    y_vis=1,
    proba_y_vis_name='P(healthy)', 
    features_vis_order = list(range(len(clf.active_features_names))),
    features_vis_name = clf.active_features_names,
    features_vis_x_axis_min = [ 29.,   0.,  94., 126.,   0.,  71.,   0.,   0.,   0.,   0.,   0., 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
    features_vis_x_axis_max = [ 77. ,   1. , 200. , 564. ,   1. , 202. ,   1. ,   6.2,   3. ,    2. ,   1. ,   1. ,   1. ,   1. ,   1. ,   1. ,   1. ,   1. , 1. ,   1. ],
    features_init_value = features_init_value,
    features_is_only_min_and_max = features_is_only_min_and_max
)
