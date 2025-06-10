import pandas as pd
import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def _cumsum_of_y_sorted_by_reversed_y_pred_proba_1(y, y_pred_prob):
    return np.cumsum([x for _, x in sorted(zip(y_pred_prob, y), reverse=True, key=lambda x: x[0])])

def experiment_auc(clf, X_train, y_train, X_test, y_test, y_class_names, res_table, visualize_confusion_matrix=False, visualize_roc_curve_fig=None, visualize_roc_curve_fig_line=None):
    clf.fit(X_train, y_train)
    y_train_pred_proba = clf.predict_proba(X_train)
    auc_train = metrics.roc_auc_score(y_train, y_train_pred_proba[:, 1]) * 100
    y_test_pred_proba = clf.predict_proba(X_test)
    #print(sorted(y_test_pred_proba[:, 1], reverse=True))
    auc_test = metrics.roc_auc_score(y_test, y_test_pred_proba[:, 1]) * 100

    #max_curve = _cumsum_of_y_sorted_by_reversed_y_pred_proba_1(y_test, y_test)
    #max_auc = np.sum(max_curve)

    res_table.append([f'{clf}', f'{auc_train:.2f}%', f'{auc_test:.2f}%'])
    
    if visualize_roc_curve_fig is not None:
        # fpr, tpr, _ = metrics.roc_curve(y_test, y_test_pred_proba[:, 1])
        # visualize_roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{clf}', line=visualize_roc_curve_fig_line))
        clf_curve = _cumsum_of_y_sorted_by_reversed_y_pred_proba_1(y_test, y_test_pred_proba[:, 1])
        visualize_roc_curve_fig.add_trace(go.Scatter(y=clf_curve, name=f'{clf}', line=visualize_roc_curve_fig_line))

                
    if visualize_confusion_matrix:
        y_pred = clf.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        cnf_df = pd.DataFrame(cnf_matrix)
        
        _, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
        sns.heatmap(cnf_df, annot=True, fmt='.2f')
        ax.xaxis.set_label_position("bottom")
        tick_marks = np.arange(len(y_class_names)) + 0.5
        plt.title(' '.join(res_table[-1]))
        plt.xticks(tick_marks, y_class_names)
        plt.yticks(tick_marks, y_class_names)
        plt.tight_layout()
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
