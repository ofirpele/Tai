import os, pickle
import prediction_lens
from pathlib import Path

#MODEL_FILENAME = 'LogisticRegression.pkl'
#MODEL_FILENAME = 'TaiXGBClassifier.pkl'
MODEL_FILENAME = 'TaiClassifier.pkl'

MODELS_DIR_PATH = (
    os.path.join(Path(__file__).resolve().parents[0], "models") + "\\"
)
filename = MODELS_DIR_PATH + MODEL_FILENAME
with open(filename, 'rb') as fp:
    clf = pickle.load(fp)

# # For Tai XGB
# features_init_value = [250,             10,       5,             5000,                10000,              0]
# features_init_value = [774,             0,       0,             7,                   0,              0]
# For full Tai
features_init_value = [217,             0,       4.44,             5000,                10000,              0]
features_is_only_min_and_max = [False] * len(clf.active_features_names)
features_is_only_min_and_max[-1] = True
prediction_lens.make_and_run_app(
    clf, 
    y_vis=0,
    proba_y_vis_name='P(ok)', 
    features_vis_order = [5, 4, 1, 0, 3, 2],
    features_vis_name = ['code length', '#comment', 'entropy', 'time since creation', 'time since last fix', 'unit test'],
    features_vis_x_axis_min = [-0.02] * len(clf.active_features_names),
    features_vis_x_axis_max = [801,          115,       7.55,            19500,                 19500,             1.02],
    features_init_value = features_init_value,
    features_is_only_min_and_max = features_is_only_min_and_max,
)
