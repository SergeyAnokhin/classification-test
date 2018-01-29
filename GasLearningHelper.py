import sqlite3
from domain import Measure
from domain import AverageData


class GasLearningHelper:
    def ReadGasData(self, fileName, startDate):
        conn = sqlite3.connect(fileName)
        c = conn.cursor()
        result = []
        for row in c.execute("SELECT date, value, comment FROM reading WHERE meterid=3 and date > date('" +
                             startDate + "') order by date"):
            meas = Measure()
            meas.date = row[0]
            meas.value = row[1]
            meas.comment = row[2]
            result.append(meas)

        c.close()
        return result

    def CreateDelta(self, data):
        last_meas = None
        deltas = []
        for row in data:
            if last_meas != None:
                delta = DeltaData()
                delta.start = last_meas
                delta.end = row
                deltas.append(delta)
            last_meas = row

#        result = {}
#        for d in deltas:
#            result[d.TimeKey] = d
#        return result
        return deltas


class DeltaData(AverageData):
    def Value(self):
        return self.end.value - self.start.value

    def __str__(self):
        return '{0} at {1}, {2}'.format(self.Value, self.TimeKey, self.Comment)
