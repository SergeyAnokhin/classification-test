class Measure:
    date = None
    value = None
    comment = None

    def __str__(self):
        return '{0} at {1}, {2}'.format(self.value, self.date, self.comment)


class AverageData:
    start: Measure = None
    end: Measure = None

    def Duration(self):
        return self.end.date - self.start.date

    def Comment(self):
        return '{0} -> {1}'.format(self.start, self.end)

    def Value(self):
        return (self.end.value + self.start.value) / 2.0

    def TimeKey(self):
        return '{0}.000Z-{1}.000Z'.format(self.start.date, self.end.date)

