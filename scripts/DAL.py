import csv
import numpy as np
import matplotlib.pyplot as plt

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

    def len_info(self, plot : bool = False) -> dict:
        """! Get len info and plot histogram
            @param plot (optional) if true plot histogram
            @return dict(min, max, mean, std, 95% interval)"""
        
        # Get lengths, mean and std
        lengths = np.array([len(text.split(" ")) for _, text in self.get()])
        mean = np.mean(lengths)
        std = np.std(lengths)
        
        # Assuming normal distribution 95% interval
        recomended = int(mean + 2*std)

        # Histogram
        if plot:
            plt.hist(lengths, bins=np.arange(min(lengths), max(lengths), 100))
            plt.title("Text length Histogram (in number of words)")
            plt.ylabel("Number of samples")
            plt.xlabel("Text length (in words)")
            plt.axvline(recomended, color = 'r')
            plt.show()
        return {"min" : np.min(lengths), "max": max(lengths), "mean": mean, "std": std, "recomended": recomended}


if __name__ == "__main__":
    data = DAL("data.csv")
    print(data.getGrouped().keys())
    print(data.len_info(True))
