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
            print('# {:25} => OK'.format(self.name))
        else:
            print('# {:25} => {:.2f}'.format(self.name, self.tstscore))


        if self.tstscore < 1:
            unmatched = [i for i, j in zip(self.Y_tst, self.Y_pred) if i != j]
            print('  - Unmatched : {}'.format(len(unmatched)))
            self.ph.array(self.Y_pred, '{:>12}', '')

        self.displayProbas()
        print('--------------------------------------------')

    def displayProbas(self):
        if not hasattr(self.clf, 'predict_proba'):
            return
        Y_proba = self.clf.predict_proba(self.X_tst)
        for proba, pred, y_tst in zip(Y_proba, self.Y_pred, self.Y_tst):
            ind_pred = self.clf.classes_.tolist().index(pred)
            if pred == y_tst:
                probability = proba[ind_pred]
                print(' -- {} {:.2f} {}'.format(pred, 
                    probability, '#' * int(round(probability * 30))))
            else:
                ind_tst = self.clf.classes_.tolist().index(y_tst)
                print(' !! {} {:.2f} != {} {:.2f}'.format(pred, 
                    proba[ind_pred], y_tst, proba[ind_tst]))
 
    # display bar : ########
    def displayProbaWithBar(self, proba):
        zipped = zip(proba, self.clf.classes_)
        for p, c in sorted(zipped, key = lambda t: t[0]):
            print('{} {:.4f} {}'.format(c, p, '#' * int(round(p * 40))))
        #self.ph.array(proba, '{:.2f}', ' | ')
