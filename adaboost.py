import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

data = None
indptr = None
indices = None
offset = 0
firstindptr = None
Y = []
for i in range(12):
    print('output_{}.dat'.format(i + 1))
    x, y = load_svmlight_file('output_{}.dat'.format(i + 1))
    print(x.indptr)
    if i == 0:
        data = x.data
        indptr = x.indptr
        firstindptr = indptr
        indices = x.indices
        offset = x.indptr[1] - x.indptr[0]
    else:
        data = np.append(data, x.data)
        indptr = np.append(indptr, firstindptr[1:] + (x.indptr[-1] * i))
        indices = np.append(indices, x.indices)
    Y.extend(y)

Y = np.array(Y)
X = csr_matrix((data, indices, indptr), dtype=float)
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X, Y, cv=12)
print(scores)

# X_train, Y_train = load_svmlight_file('output_1.dat')
# X_test, Y_test = load_svmlight_file('output_2.dat')
# print(X_train[0])
# print(Y_train)
# clf = AdaBoostClassifier(n_estimators=100)
# clf.fit(X_train, Y_train)
# Y_pred = clf.predict(X_test)
# accu = accuracy_score(Y_test, Y_pred)
# print(accu)
