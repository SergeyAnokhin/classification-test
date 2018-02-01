import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
from ClassifierResults import ClassifierResults
from PredictionResults import PredictionResults
from printHelper import printHelper

classifiers = {
        "Neural Net(Adam, a.1)":     
            MLPClassifier(alpha=.1, max_iter=1000, solver='adam'), # , verbose=10
        "Neural Net(Adam, a.01)":     
            MLPClassifier(alpha=.01, max_iter=500, solver='adam'), # , verbose=10
        "Neural Net(Adam)":     
            MLPClassifier(alpha=1, max_iter=500, solver='adam'), # , verbose=10
        "Naive Bayes":          
            GaussianNB(),
        "RBF SVM":              
            SVC(gamma=2, C=1),
        # "Neural Net(Sgd)":      
        #     MLPClassifier(alpha=1e-2, max_iter=1000, solver='sgd'), # , verbose=10
        "Neural Net(Sgd, tol)":      
            MLPClassifier(alpha=1e-2, tol=1e-4, max_iter=500, solver='sgd', random_state=1,
                    learning_rate_init=.1), # , verbose=10
        "Neural Net(Adam)":     
            MLPClassifier(alpha=1, max_iter=500, solver='adam'), # , verbose=10
        #"Linear SVM":           SVC(kernel="linear", C=0.025),
        #"Nearest Neighbors":    
        #    KNeighborsClassifier(3),
        "Gaussian Process":     
            GaussianProcessClassifier(1.0 * RBF(1.0)),
        # "Random Forest":        
        #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        "Decision Tree":        
            DecisionTreeClassifier(max_depth=5),
        # "AdaBoost":             
        #     AdaBoostClassifier(),
        #"QDA":                  QuadraticDiscriminantAnalysis(),
    }
# print(classifiers)
hall0Pir = ('hall0Pir', 0)
hall0door = ('hall0door', 1)
cameraFront = ('cameraFront', 2)
cameraBack = ('cameraBack', 3)
corridor1Pir = ('corridor1Pir', 4)
ph = printHelper()

ctx = HomeEventsContext()
# ALRT Front
ctx.AddSequence('ALRT', {
    0 : corridor1Pir,
    1 : [hall0Pir, hall0door],
    9 : cameraFront
})
# ALRT Front Fast
ctx.AddSequence('ALRT', {
    0 : [corridor1Pir, hall0Pir, hall0door],
    5 : cameraFront
})

# ALRT back
ctx.AddSequence('ALRT', {
    0 : corridor1Pir,
    1 : hall0Pir,
    9 : cameraBack
})
# ALRT Front and back
ctx.AddSequence('ALRT', {
    0 : corridor1Pir,
    1 : hall0Pir,
    8 : cameraBack,
    9 : cameraFront
})
# father
ctx.AddSequence('MSTR', {
    0 : corridor1Pir,
    5 : [hall0Pir, hall0door],
    6 : cameraFront
})
# mather & childrens
ctx.AddSequence('MRCH', {
    0 : corridor1Pir,
    3 : [hall0Pir, hall0door],
    5 : cameraFront,
    9 : cameraFront
})
# kat
ctx.AddSequence('ANML', {
    0 : corridor1Pir,
})
# kat
ctx.AddSequence('ANML', {
    0 : hall0Pir,
})

X = ctx.getX()
Y = ctx.getY()

print('##############################################')
print('################## FIT MODEL #################')
print('##############################################')

for c in classifiers:
    clf = classifiers[c]
    clfRes = ClassifierResults()
    clfRes.fit(clf, X, Y)
    clfRes.Display(c)

print('##############################################')
print('############## TEST SEQUENCES ################')
print('##############################################')

tst = HomeEventsContext()
# tst.AddSequence('ANML', {
#     0 : hall0Pir,
# })

tst.AddSequence('ALRT', {
    0 : corridor1Pir,
    1 : [hall0Pir, hall0door],
    9 : cameraFront
})

# tst.AddSequence('ALRT', {
#     0 : [corridor1Pir, hall0Pir, hall0door],
#     9 : cameraFront
# })

tst.AddSequence('ALRT', {
     0 : [corridor1Pir, hall0Pir, hall0door],
     5 : cameraFront
 })

# mather & childrens
tst.AddSequence('MRCH', {
    0 : corridor1Pir,
    3 : [hall0Pir, hall0door],
    4 : cameraFront,
    9 : cameraFront
})


X_tst = tst.getX()
Y_tst = tst.getY()
print('Correct Classes : ')
ph.array(Y_tst, '{:>12}', ',')

for c in classifiers:
    res = PredictionResults()
    clf = classifiers[c]
    res.predict(clf, c, X_tst, Y_tst)
    res.displayResults()

#X = StandardScaler().fit_transform(X)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
