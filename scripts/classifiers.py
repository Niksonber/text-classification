import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras import backend as K
from tensorflow.keras.utils import to_categorical

"""! classifier class"""
class Classsifier:
    def __init__(self, name, preprocessors, model, rseed = 42) -> None:
        self._name = name
        self._preprocessors = preprocessors
        self._model = model
        self.rseed = rseed

        np.random.seed(rseed)

    def pre_process(self, x):
        for preprocessor in self._preprocessors:
            x = preprocessor.process(x)
        return x
    
    def train(self, x, y, proportion = 0.3):
        print("-----------------------------------------")
        print("Training: " + self._name, end='\n\n')

        #preprocess
        x = self.pre_process(x)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=proportion, random_state=self.rseed)
        print("Total data for training: {}, For testing: {} \n".format(len(y_train), len(y_test)))
        
        #train model
        self._model.fit(x_train, to_categorical(y_train), epochs=5)

        # Evaluate model
        y_pred = np.argmax(self._model.predict(x_test), axis=-1)
        cm = confusion_matrix(y_test, y_pred)
        print(classification_report(y_test, y_pred))
        print("Confusion matrix: \n\n{}".format(cm))

if __name__ == "__main__":
    K.clear_session()
    