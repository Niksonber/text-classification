import csv

class DAL:
    """ Class that provide Data abstraction layer, thus provide free acess to data"""
    
    def __init__(self, filename:str)-> None:
        """! @brief Constructor of DAL
        @param filename csv datebase path
        """
        self._filename = filename
        self._data = {}
        self._info = { "len_info" : self.len_info}
    
    def get(self) -> list:
        """! Get raw data
        @return list of rows in csv [label, text]"""

        with open(self._filename, 'rt') as f:
            # remove collums titles
            return list(csv.reader(f))[1:]
    
    def getGrouped(self) -> dict:
        """! Get data grouped by label"""
        raw = self.get()
        labels = set([row[0] for row in raw])
        data = {label:[text] for label in labels for c, text in raw if c == label}
        return data

    def info(self) -> dict:
        """! Get info of data """

    def len_info(self) -> dict:
        """! Get len info"""


if __name__ == "__main__":
    data = DAL("data.csv")
    print(data.getGrouped().keys())
