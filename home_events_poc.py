import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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


class HomeEventsContext:
    images_X = []
    images_Y = []

    def AddSequence(self, name, seq):
        self.images_Y.append(name)
        image = []
        for i in range(0, 10):
            image_line = [0, 0, 0, 0, 0, 0]
            if i in seq:
                events = seq[i]
                if isinstance(events, list):
                    for event in events:
                        self.ApplyEvent(image_line, event)
                else:
                    self.ApplyEvent(image_line, events)
                    
            image.append(image_line)
        self.images_X.append(image)

    def ApplyEvent(self, line, event):
        i = event[1]
        line[i] = 1
