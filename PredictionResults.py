from printHelper import printHelper

class PredictionResults:
    clf = None
    tstscore = 0
    Y_pred = None
    X_tst = None
    Y_tst = None
    name = ''
    ph = printHelper()

    def predict(self, clf, name, X_tst, Y_tst):
        self.X_tst = X_tst
        self.Y_tst = Y_tst
        self.clf = clf
        self.name = name
        self.tstscore = clf.score(X_tst, Y_tst)
        self.Y_pred = clf.predict(X_tst)

    def displayResults(self):
        isOk = ' '
        if self.tstscore == 1:
            isOk = ' OK'
        print('# {:25} => {} {:.2f}'.format(self.name, isOk, self.tstscore))


        if self.tstscore < 1:
            unmatched = [i for i, j in zip(self.Y_tst, self.Y_pred) if i != j]
            print('  - Unmatched : {}'.format(len(unmatched)))
            self.ph.array(self.Y_pred, '{:>12}', '')
        if hasattr(self.clf, 'predict_proba'):
            Y_proba = self.clf.predict_proba(self.X_tst)
            self.ph.array(self.clf.classes_, '{:<7}', '')
            for i in Y_proba:
                self.ph.array(i, '{:.2f}', ' | ')
        print('-------------------------------------------')

