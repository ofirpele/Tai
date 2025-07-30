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

import os
from pathlib import Path


########################################################################################################################
categorical_features_names = ['cp', 'restecg', 'slope']
y_name = ['condition']

RAW_DATA_DIR_PATH = (
    os.path.join(Path(__file__).resolve().parents[0], "data") + "\\"
)
data_df = pd.read_csv(RAW_DATA_DIR_PATH + 'heart_cleveland_upload.csv')

y = data_df[y_name].to_numpy()
y = y.squeeze()
y_class_names = ['Heart Condition', 'Healthy']

data_df = data_df.drop(y_name, axis=1)
data_df = pd.get_dummies(data_df, columns=categorical_features_names)
features_names = data_df.columns.tolist()

X = data_df.to_numpy()

# X = X.astype(float)
# print(repr(np.min(X, axis=0))[6:-1])
# print(repr(np.mean(X, axis=0))[6:-1])
# print(repr(np.max(X, axis=0))[6:-1])
# print('[ ', end=' ')
# for c in range(X.shape[1]):
#     num = X.shape[0]
#     num_unique = len(np.unique(X[:, c]))
#     #print(f'{data_df.columns[c]} {num_unique=} ({num_unique*100/num:.2f}%)')
#     if num_unique == 2:
#         print ('True,', end=' ')
#     else:
#         print('False,', end=' ')
# print(']')
# exit()

# X = X[:30,:]
# X[:,1] = 1
# X[:,3] = 5
# y = np.random.randint(2, size=(30))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=42)

print(f'#elements y_train: {len(y_train)}')
print(f'#elements y_test : {len(y_test)}')
print(f'#{y_class_names[1]} in y_train: {sum(y_train).item()} ({sum(y_train).item()*100/len(y_train):.2f}%)')
print(f'#{y_class_names[1]} in y_test : {sum(y_test).item()} ({sum(y_test).item()*100/len(y_test):.2f}%)')
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
            text='Patients sorted by predicted probability of illness (high first)'
        )
    ),
    yaxis=dict(
        title=dict(
            # If we go back to TPR: (y_pred=fix && y=fix)/(#fix)
            text='Cumaltive sum of ill patients'
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

# clf = TaiLinearClassifier(linear_classifier=LogisticRegression(**logistic_regression_params_dict))
# experiment_auc(
#     clf,
#     **experiment_shared_params,
#     visualize_roc_curve_fig_line = dict(color='blue')
# )
# print(res_table[-1])

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
    [classifier_unit.TFL(num_epochs=1, catboost_init_dict=catboost_init_dict)],
    [classifier_unit.XGB(random_state=42), classifier_unit.TFL(num_epochs=1, catboost_init_dict=catboost_init_dict)],
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
    visualize_roc_curve_fig_line['dash'] = 'dash'
    experiment_auc(
        TaiClassifier(with_constraints_from_logistic_regression=False, features_names=features_names, classifiers=tai_classifiers),
        **experiment_shared_params,
        visualize_roc_curve_fig_line = visualize_roc_curve_fig_line
    )
    visualize_roc_curve_fig_line.pop('dash')
    clf = TaiClassifier(with_constraints_from_logistic_regression=True, logistic_regression_params_dict=logistic_regression_params_dict, features_names=features_names, classifiers=tai_classifiers)
    experiment_auc(
        clf,
        **experiment_shared_params,
        visualize_roc_curve_fig_line = visualize_roc_curve_fig_line
    )
    
# open_mode = 'wb+'
# import os, pickle
# MODEL_FILENAME = 'TaiClassifier_TFL.pkl'
# with open(MODEL_FILENAME, open_mode) as fp:
#     pickle.dump(clf, fp, pickle.HIGHEST_PROTOCOL)

ccol = f'Classifier{"":45}'
df_res = pd.DataFrame(res_table, columns=[ccol, 'Train AUC%', 'Test AUC%'])
print()
print(df_res.to_string(index=False, formatters={ccol: lambda x: f'{x:<55}'}))

roc_curve_fig.show()
########################################################################################################################
