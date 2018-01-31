import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timeit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from HomeEventsContext import HomeEventsContext

classifiers = {
        "RBF SVM":              
            SVC(gamma=2, C=1),
        "Decision Tree":        
            DecisionTreeClassifier(max_depth=5),
        "Random Forest":        
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        "Neural Net(Adam)":     
            MLPClassifier(alpha=1, max_iter=500, solver='adam'), # , verbose=10
        "Neural Net(Adam, a0.1)":     
            MLPClassifier(alpha=.1, max_iter=500, solver='adam'), # , verbose=10
        "Neural Net(Adam, a0.01)":     
            MLPClassifier(alpha=.01, max_iter=500, solver='adam'), # , verbose=10
        "Neural Net(Adam)":     
            MLPClassifier(alpha=1, max_iter=500, solver='adam'), # , verbose=10
        "Naive Bayes":          
            GaussianNB(),
        "Neural Net(Sgd)":      
            MLPClassifier(alpha=1, max_iter=500, solver='sgd'), # , verbose=10
        #"Linear SVM":           SVC(kernel="linear", C=0.025),
        "Nearest Neighbors":    
            KNeighborsClassifier(3),
        "Gaussian Process":     
            GaussianProcessClassifier(1.0 * RBF(1.0)),
        "AdaBoost":             
            AdaBoostClassifier(),
        #"QDA":                  QuadraticDiscriminantAnalysis(),
    }
# print(classifiers)
hall0Pir = ('hall0Pir', 0)
hall0door = ('hall0door', 1)
cameraFront = ('cameraFront', 2)
cameraBack = ('cameraBack', 3)
corridor1Pir = ('corridor1Pir', 4)

ctx = HomeEventsContext()

ctx.AddSequence('alert', {
    0 : corridor1Pir,
    1 : [hall0Pir, hall0door],
    9 : cameraFront
})

ctx.AddSequence('alert', {
    0 : corridor1Pir,
    1 : hall0Pir,
    9 : cameraBack
})

ctx.AddSequence('alert', {
    0 : corridor1Pir,
    1 : hall0Pir,
    8 : cameraBack,
    9 : cameraFront
})

ctx.AddSequence('master_come', {
    0 : corridor1Pir,
    5 : [hall0Pir, hall0door],
    6 : cameraFront
})

ctx.AddSequence('masterChild_come', {
    0 : corridor1Pir,
    5 : [hall0Pir, hall0door],
    9 : cameraFront
})

ctx.AddSequence('anomalie', {
    0 : corridor1Pir,
})

ctx.AddSequence('anomalie', {
    0 : hall0Pir,
})

#for i, n in zip(ctx.images_X, ctx.images_Y):
#    display(n, i)

X = ctx.getX()
Y = ctx.getY()
ctx.clean()

tst = HomeEventsContext()
tst.AddSequence('anomalie', {
    0 : hall0Pir,
})

# X_pred1 = [[1, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0]]
X_tst = tst.getX()
Y_tst = tst.getY()

for c in classifiers:

    clf = classifiers[c]
    start_time = timeit.default_timer()
    clf.fit(X, Y)
    elapsed = timeit.default_timer() - start_time
    selfscore = clf.score(X, Y)
    if selfscore < 1:
        print('# {:25} => {:.2f} ({:.2f} sec)'.format(c, selfscore, elapsed))
        Y_pred = clf.predict(X_tst)
        matched = [i for i, j in zip(Y_tst, Y_pred) if i != j]
        print('Unmatched : {}'.format(len(matched)))
        print(Y_pred)
    #else:
    #    print('# {:25} => OK {:.2f} ({:.2f} sec)'.format(c, selfscore, elapsed))


for c in classifiers:

    clf = classifiers[c]
    start_time = timeit.default_timer()
    clf.fit(X, Y)
    elapsed = timeit.default_timer() - start_time
    selfscore = clf.score(X, Y)
    if selfscore < 1:
        print('# {:25} => {:.2f} ({:.2f} sec)'.format(c, selfscore, elapsed))
        #pred_Y = clf.predict(X)
        #matched = [i for i, j in zip(Y, pred_Y) if i != j]
        #display(len(matched))
        #display(pred_Y)
    else:
        print('# {:25} => OK {:.2f} ({:.2f} sec)'.format(c, selfscore, elapsed))

#X = StandardScaler().fit_transform(X)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
