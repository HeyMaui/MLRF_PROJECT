import joblib
import logging
from sklearn.metrics import accuracy_score


class Classifier:

    def __init__(self, model=None):
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def fit_classifier(self, train_descriptors: list, train_labels: list):
        """
        Fit the classifier

        Parameters
        ----------
        train_descriptors : list
            List of train descriptors

        train_labels : list
            List of train labels
        """
        if self.model is None:
            self.logger.error("No model found, please provide a model from sklearn")
            return
        self.logger.info("Fitting data on classifier, this may take a while...")
        self.model.fit(train_descriptors, train_labels)
        self.logger.info("Training done")

    def predict(self, test_descriptors: list):
        """
        Predict the class of the data

        Parameters
        ----------
        test_descriptors : list
            List of test descriptors

        Returns
        -------
        list
            List of predictions
        """
        if self.model is None:
            self.logger.error("No model found, please provide a model from sklearn")
            return
        return self.model.predict(test_descriptors)

    def evaluate(self, test_descriptors: list, test_labels: list):
        """
        Evaluate the model

        Parameters
        ----------
        test_descriptors : list
            List of test descriptors

        test_labels : list
            List of test labels

        Returns
        -------
        float
            Accuracy
        """
        if self.model is None:
            self.logger.error("No model found, please provide a model from sklearn")
            return
        y_pred = self.predict(test_descriptors)
        return accuracy_score(test_labels, y_pred)

    def save_model(self, filename):
        """
        Save the model

        Parameters
        ----------
        filename : str
            Name of the file
        """
        if self.model is None:
            self.logger.error("No model found, please provide a model from sklearn")
            return
        joblib.dump(self.model, "models/" + filename)

    def load_model(self, filename):
        """
        Load the model

        Parameters
        ----------
        filename : str
            Name of the file
        """
        self.model = joblib.load("models/" + filename)
