import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.base import BaseEstimator
from typing import Union, Callable
from joblib import load


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)

    class_names = ['0', '1']

    plt.figure(figsize=(8, 6), facecolor='#F4F4F4')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True classes')
    plt.xlabel('Predicted classes')
    plt.tight_layout()
    plt.show()
    
def my_classification_report(base_search_cv: BaseSearchCV, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray):
    print("Best score: {:.3f}".format(base_search_cv.best_score_))
    print("Best params: {}".format(base_search_cv.best_params_))
    pred_devtrain = base_search_cv.predict(X_train)
    pred_val = base_search_cv.predict(X_val)
    print('Cross report on train set:')
    cross_report(estimator=base_search_cv.best_estimator_, X=X_train, y=y_train)
    print("Classification report on train set:\n {}".format(classification_report(y_true=y_train, y_pred=pred_devtrain)));     
    print("Classification report on val set:\n {}".format(classification_report(y_true=y_val, y_pred=pred_val)));
    plot_confusion_matrix(y_true=y_val, y_pred=pred_val)   

def my_classification_report_light(estimator: BaseEstimator, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray):
    pred_devtrain = estimator.predict(X_train)
    pred_val = estimator.predict(X_val)
    print("Classification report on train set:\n {}".format(classification_report(y_true=y_train, y_pred=pred_devtrain)));
    print("Classification report on val set:\n {}".format(classification_report(y_true=y_val, y_pred=pred_val)));
    plot_confusion_matrix(y_true=y_val, y_pred=pred_val)

def search_report(filename: str, X_val: np.ndarray, y_val:np.ndarray):
    loaded = load(filename)
    model = loaded['model']
    x = loaded['x']
    y = loaded['y']
    my_classification_report(base_search_cv=model, X_train=x, X_val=X_val, y_train=y, y_val=y_val)

def plot_roc_from_tuned(filename: str, label: str, X: np.ndarray, y: np.ndarray):
    model = load(filename)['model']
    fpr, tpr, ts = roc_curve(y_true=y, y_score=model.best_estimator_.predict_proba(X)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=label+' (area={:.3f},cvs={:.3f})'.format(roc_auc,model.best_score_))
    
def plot_pr_from_tuned(filename: str, label: str, X: np.ndarray, y: np.ndarray):
    model = load(filename)['model']
    pr, rc, ts = precision_recall_curve(y_true=y, probas_pred=model.best_estimator_.predict_proba(X)[:,1])
    plt.plot(rc, pr, label=label)
    
def cross_report(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv=5):
    scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    results = cross_validate(estimator=estimator, scoring=scoring, X=X, y=y, cv=cv)
    print('\tAccuracy:\t{:.4f}'.format(results['test_accuracy'].mean()))
    print('\tF1-score:\t{:.4f}'.format(results['test_f1'].mean()))
    print('\tPrecision:\t{:.4f}'.format(results['test_precision'].mean()))
    print('\tRecall:\t\t{:.4f}'.format(results['test_recall'].mean()))
    print('\tROC_AUC:\t{:.4f}'.format(results['test_roc_auc'].mean()))
    
#define a function for plotting every numerical feature by passing different values to common_norm parameter
def histplotNumericalFeatures(features: pd.DataFrame, target: Union[pd.Series, np.ndarray], common_norm: bool = False):

    df = features.copy()
    df['loan_status'] = target
    
    #initizalize subplots
    fig, axs = plt.subplots(4,3, figsize=[20,15])
    fig.set_facecolor('#F4F4F4')

    #set title
    if common_norm:
        fig.suptitle("Distribution of numerical features differentiated by loan status.")
    else:
        fig.suptitle("Distribution of numerical features differentiated by loan status. \nIn each graph, the two distributions are independently normalized.")
    #plt.subplots_adjust(top=0.7)
    #plt.title('\n', y=0.9)

    #loan_int_rate
    if 'loan_int_rate' in features: 
        sns.histplot(data=df, x='loan_int_rate', hue='loan_status', stat='proportion', discrete=False, common_norm=common_norm, multiple='layer', element='step', fill=False, ax=axs[0,0])

    #loan_percent_income
    if 'loan_percent_income' in features: 
        sns.histplot(data=df, x='loan_percent_income', hue='loan_status', stat='proportion', discrete=False, common_norm=common_norm, multiple='layer', element='step', fill=False, ax=axs[0,1])

    #cb_person_cred_hist_length
    if 'cb_person_cred_hist_length' in features: 
        sns.histplot(data=df, x='cb_person_cred_hist_length', hue='loan_status', stat='proportion', discrete=True, common_norm=common_norm, multiple='layer', element='step', fill=False, ax=axs[0,2])

    #person_age full
    if 'person_age' in features: 
        sns.histplot(data=df, x='person_age', hue='loan_status', stat='proportion', discrete=True, common_norm=False, multiple='layer', element='step', fill=False, ax=axs[1,0])

    #person_age truncated
    if 'person_age' in features:
        sns.histplot(data=df[df.person_age <= 70], x='person_age', hue='loan_status', stat='proportion', discrete=True, common_norm=common_norm, multiple='layer', element='step', fill=False, ax=axs[1,1])
        axs[1,1].set_xlabel("person_age (trunc. at 70)")

    #person_income full
    if 'person_income' in features:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        axs[2,0].xaxis.set_major_formatter(formatter)
        sns.histplot(data=df, x='person_income', hue='loan_status', stat='proportion', discrete=False, common_norm=common_norm, multiple='layer', element='step', fill=False, ax=axs[2,0])

    #person_income trucanted
    if 'person_income' in features:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        axs[2,1].xaxis.set_major_formatter(formatter)
        sns.histplot(data=df[df.person_income < 3e5], x='person_income', hue='loan_status', stat='proportion', discrete=False, common_norm=common_norm, multiple='layer', element='step', fill=False, ax=axs[2,1])
    axs[2,1].set_xlabel("person_income (trunc. at 3e5)")

    #person_emp_length full
    if 'person_emp_length' in features:
        sns.histplot(data=df, x='person_emp_length', hue='loan_status', stat='proportion', discrete=True, common_norm=common_norm, multiple='layer', element='step', fill=False, ax=axs[3,0])

    #person_emp_length truncated
    if 'person_emp_length' in features:
        sns.histplot(data=df[df.person_emp_length <= 40], x='person_emp_length', hue='loan_status', stat='proportion', discrete=True, common_norm=common_norm, multiple='layer', element='step', fill=False, ax=axs[3,1])
        axs[3,1].set_xlabel("person_emp_length (trunc. at 40)")

    #delete empty subplots
    fig.delaxes(axs[1,2])
    fig.delaxes(axs[2,2])
    fig.delaxes(axs[3,2])

    #increase space between subplots
    plt.tight_layout()

    plt.show()

def prepare_voting_classifier(estimators: list[BaseEstimator], estimators_labels: list[str], weigth_score: Callable[[np.ndarray, np.ndarray], float], x: np.ndarray, y_true: np.ndarray):
    estimators_output = [(label, estimator) for label, estimator in zip(estimators_labels, estimators)]
    weigths_output = [weigth_score(y_true, estimator.predict(x)) for estimator in estimators]
    return estimators_output, weigths_output