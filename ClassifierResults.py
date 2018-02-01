import timeit

class ClassifierResults:
    fit_elapsed = 0
    score = 0
    loss = 0
    X = []
    Y = []
    clf = None

    # def __new__(self, clf):
    #     self.clf = clf 
    #     return self

    def fit(self, clf, x, y):
        self.X = x
        self.Y = y
        self.clf = clf 

        start_time = timeit.default_timer()
        clf.fit(x, y)
        self.fit_elapsed = timeit.default_timer() - start_time
        self.score = clf.score(x, y)
        if hasattr(clf, 'loss_'):
            self.loss = clf.loss_
        return

    def Display(self, name):
        self.printTotal(name)
        if self.score < 1:
            pred_Y = self.clf.predict(self.X)
            unmatched = [i for i, j in zip(self.Y, pred_Y) if i != j]
            print(unmatched)

    def printTotal(self, name):
        isOk = ' '
        if self.score = 1
            isOk = ' OK'
        print('# {:22} =>{} {:.2f} +-{:.2f} ({:.2f} sec)'
            .format(name, isOk, self.score, self.loss, self.fit_elapsed))

