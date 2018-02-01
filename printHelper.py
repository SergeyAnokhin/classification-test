class printHelper:

    def returnParamsAsArray(self, array, fmt):
        return fmt.format(*array)

    def paramsAsArray(self, array, fmt):
        print(self.returnParamsAsArray(array, fmt))

    def array(self, array, itemFormat, splitter):
        s = self.listToFormattedString(array, itemFormat, splitter)
        print(s)

    def arrayIntro(self, array, intro, itemFormat, splitter):
        s = intro + self.listToFormattedString(array, itemFormat, splitter)
        print(s)

    def listToFormattedString(self, array, itemFormat, splitter):
        formatted_list = [itemFormat for item in array] 
        s = splitter.join(formatted_list)
        return s.format(*array)
