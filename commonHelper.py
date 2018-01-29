import json
import sklearn.preprocessing as preprocessing
import numpy as np

class CommonHelper:
    def ReadJson(self, filename):
        return json.load(open(filename))


    def WriteJson(self, filename, jsondata):
        json.dump(jsondata, open(filename, 'w'))

    def FilterByTimeKey(self, data, timeKey):
        found = filter(lambda x: x.TimeKey == timeKey, data)
        return found

    def Normalize(self, list1d):
        size = len(list1d)
        array = np.asarray(list1d)
        result = preprocessing.MinMaxScaler().fit_transform(array.reshape(-1, 1)).reshape(size)
        return result

