from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mlxtend.data import wine_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, \
        log_loss, \
        classification_report, \
        confusion_matrix, \
        roc_auc_score,\
        average_precision_score,\
        auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef, auc

#iRec1 = 'fctd.csv'
#iRec2 = 'fconjTRD.csv'
#iRec3 = 'fpsc.csv'

"""ConjointTRD"""
iRecConjoint = 'SC_PseAAC_General.csv'
D = pd.read_csv(iRecConjoint) #header=None)  # Using pandas
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values
#X_train, X_test, y_train, y_test= train_test_split(Xtrd, ytrd, stratify=ytrd, test_size=0.3,random_state=1)
#knn = KNeighborsClassifier(n_neighbors=10)
svm=SVC()
from sklearn.model_selection import StratifiedKFold,KFold
cv = KFold(n_splits=5, random_state=None, shuffle=True)

for (train_index, test_index) in cv.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    svm.fit(X_train, y_train)
sfs1 = SFS(estimator=svm,k_features=(3, 20),forward=True,floating=False,scoring='accuracy',cv=5)
#sfs1 = SFS(estimator=knn,k_features=(3, 10),forward=True,floating=False,scoring='accuracy',cv=5)
pipe = make_pipeline(StandardScaler(), sfs1)
pipe.fit(X_train, y_train)
print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print('all subsets:\n', sfs1.subsets_)

fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')

plt.ylim([0.50, 0.100])
plt.title('SFFS-SVM')
plt.grid()
plt.show()

Cs = [0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid_svm = {'C': Cs, 'gamma' : gammas}
grid_mse = GridSearchCV(estimator = svm, param_grid=param_grid_svm, scoring = 'accuracy', cv = 5, verbose = 1)

grid_mse.fit(X_train, y_train)
print("Best parameters found: ",grid_mse.best_params_)
print("Accuracy for training: ", np.sqrt(np.abs(grid_mse.best_score_)))

pred = grid_mse.predict(X_test)
print("Accuracy for test dataset: {}".format(accuracy_score(y_test, pred)))
