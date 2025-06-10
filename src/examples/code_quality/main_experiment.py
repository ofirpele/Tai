import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from TaiLinearClassifier import TaiLinearClassifier
from TaiClassifier import TaiClassifier

import plotly.graph_objects as go

import classifier_unit

from experiment_auc import experiment_auc
########################################################################################################################

from .config import MERGED_DATA_FILE
from examples.code_quality.CategoricalFeaturesUtils import CategoricalFeaturesUtils

SAVE_TEST_WITH_GIT_LABEL = False


########################################################################################################################
features_names = ['class_name', 'code_len', 'calc_comments_number', 'entropy', 'time_from_intro', 'time_from_last_fix', 'has_unit_test']
if SAVE_TEST_WITH_GIT_LABEL:
    features_names.append('git_label_serial_number')
y_name = ['num_of_fixes']

data_df = pd.read_csv(MERGED_DATA_FILE)

X = data_df[features_names].to_numpy()

y = data_df[y_name].to_numpy()
y = y.squeeze()
y = (y > 0)
y_class_names = ['No Fixes', 'Fixes']

cfu = CategoricalFeaturesUtils(X, 0)
X, y = cfu.samples_one_in_category(X, y, location='first')
# remove class_name
if not SAVE_TEST_WITH_GIT_LABEL:
    assert(features_names[0] == 'class_name')
    X = X[:, 1:]
    features_names = features_names[1:]

# X = X[:30,:]
# X[:,1] = 1
# X[:,3] = 5
# y = np.random.randint(2, size=(30))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

if SAVE_TEST_WITH_GIT_LABEL:
    np.savetxt('X_test.csv', X_test, delimiter=',', header=','.join(features_names), fmt='%s', comments='')
    np.savetxt('y_test.csv', y_test, delimiter=',', header='fixes>0', fmt='%s', comments='')
    exit()


print(f'#elements y_train: {len(y_train)}')
print(f'#elements y_test : {len(y_test)}')
print(f'#fixes y_train: {sum(y_train).item()} ({sum(y_train).item()*100/len(y_train):.2f}%)')
print(f'#fixes y_test : {sum(y_test).item()} ({sum(y_test).item()*100/len(y_test):.2f}%)')

# for c in range(X.shape[1]):
#     num = X.shape[0]
#     num_unique = len(np.unique(X[:, c]))
#     print(f'{X_names[c]} {num_unique=} ({num_unique*100/num:.2f}%)')
# print()
########################################################################################################################


########################################################################################################################
logistic_regression_params_dict = {
    'penalty': None, 
    'class_weight': 'balanced'
}
########################################################################################################################


########################################################################################################################
roc_curve_fig = go.Figure()
roc_curve_fig.update_layout(
    title=dict(
        text=''
    ),
    xaxis=dict(
        title=dict(
            # If we go back to FPR: (y_pred=fix && y=no fix)/(#no fix)
            text='Classes sorted by predicted probability of fix (high first)'
        )
    ),
    yaxis=dict(
        title=dict(
            # If we go back to TPR: (y_pred=fix && y=fix)/(#fix)
            text='Cumaltive sum of real fixes of classes'
        )
    ),
)

res_table = []
experiment_shared_params = {
    'X_train' : X_train,
    'y_train' : y_train, 
    'X_test' : X_test, 
    'y_test' : y_test,
    'y_class_names' : y_class_names,
    'res_table' : res_table,
    'visualize_roc_curve_fig' : roc_curve_fig
}

# tai_classifiers = [
#     classifier_unit.LogisticRegression(**logistic_regression_params_dict),
# ]
# clf = TaiClassifier(with_constraints_from_logistic_regression=True, logistic_regression_params_dict=logistic_regression_params_dict, features_names=features_names, classifiers=tai_classifiers)
# experiment_auc(
#     clf,
#     **experiment_shared_params,
#     visualize_roc_curve_fig_line = dict(dash='dot', color='yellow')
# )

clf = TaiLinearClassifier(linear_classifier=LogisticRegression(**logistic_regression_params_dict))
experiment_auc(
    clf,
    **experiment_shared_params,
    visualize_roc_curve_fig_line = dict(color='blue')
)

catboost_init_dict = {
    'random_seed' : 42,
    'allow_writing_files' : False,
    'silent' : True
}

tai_classifiers_arr = [
    #[classifier_unit.CatBoost(**catboost_init_dict)],
    [classifier_unit.XGB(random_state=42)],
    [classifier_unit.TFL(catboost_init_dict=catboost_init_dict)],
    [classifier_unit.XGB(random_state=42), classifier_unit.TFL(catboost_init_dict=catboost_init_dict)],
]
tai_classifiers_visualize_roc_curve_fig_line_colors = [
    'purple',
    'orange',
    'green',
]

for tai_classifiers, tai_classifiers_visualize_roc_curve_fig_line_color in zip(tai_classifiers_arr, tai_classifiers_visualize_roc_curve_fig_line_colors):
    visualize_roc_curve_fig_line = {
        'color' : tai_classifiers_visualize_roc_curve_fig_line_color,
    }
    clf = TaiClassifier(with_constraints_from_logistic_regression=True, logistic_regression_params_dict=logistic_regression_params_dict, features_names=features_names, classifiers=tai_classifiers)
    experiment_auc(
        clf,
        **experiment_shared_params,
        visualize_roc_curve_fig_line = visualize_roc_curve_fig_line
    )
    visualize_roc_curve_fig_line['dash'] = 'dash'
    experiment_auc(
        TaiClassifier(with_constraints_from_logistic_regression=False, features_names=features_names, classifiers=tai_classifiers),
        **experiment_shared_params,
        visualize_roc_curve_fig_line = visualize_roc_curve_fig_line
    )
    
# open_mode = 'wb+'
# import os, pickle
# #MODEL_FILENAME = 'LogisticRegression.pkl'
# #MODEL_FILENAME = 'TaiXGBClassifier.pkl'
# MODEL_FILENAME = 'TaiClassifier.pkl'
# with open(filename, open_mode) as fp:
#     pickle.dump(clf, fp, pickle.HIGHEST_PROTOCOL)

df_res = pd.DataFrame(res_table, columns=['Classifier', 'Train AUC%', 'Test AUC%'])
print(df_res.to_string(index=False))

roc_curve_fig.show()

# exit()

########################################################################################################################
