import logging
from math import e
import cv2
import joblib
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from skimage.feature import hog
import pickle
from sklearn.metrics import accuracy_score
import copy

from cifar10_classifier.config import Config


class Cifar10Classifier:
    def __init__(self, feature_extractor="HOG", model=None):
        """
        Initialize the classifier

        Parameters
        ----------
        feature_extractor : str
            Feature extractor to use
        classifier : str
            Classifier to use
        model : object
            Model to use
        """
        self.feature_extractor = feature_extractor
        self.train_descriptors = []
        self.test_descriptors = []
        self.train_labels = []
        self.test_labels = []
        self.model = model
        self.kmeans = None
        self.pca_transform = None
        self.mean = None
        self.logger = logging.getLogger(__name__)

    def load_data(self, path):
        """
        Load the data from the path

        Parameters
        ----------
        path : str
            Path to the data

        Returns
        -------
        list
            List of images
        list
            List of labels
        """
        data = []
        labels = []
        self.logger.info("Loading data from: " + path)
        with open(path, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        for img_flat in dict[b"data"]:
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))
            data.append(img)
        for label in dict[b"labels"]:
            labels.append(label)
        self.logger.info("Data loaded")
        return data, labels

    def extract(self, data, labels, train=True):
        """
        Extract features from the train data

        Parameters
        ----------
        data : list
            List of images

        labels : list
            List of labels

        train : bool
            If the data is train data

        Returns
        -------
        list
            List of features
        """
        descriptors = []
        self.logger.info("Extracting features, this may take a while...")
        if self.feature_extractor == "flatten":
            descriptors = [img.flatten() for img in data]
        elif self.feature_extractor == "HOG":
            for img in data:
                desc, _ = hog(
                    img,
                    visualize=True,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    orientations=8,
                    channel_axis=-1,
                )
                if desc is None:
                    desc = np.zeros((0, 128), dtype="uint8")
                descriptors.append(desc)
        elif self.feature_extractor == "SIFT":
            sift = cv2.SIFT_create()
            for i, img in enumerate(data):
                _, desc = sift.detectAndCompute(img, None)
                if desc is None:
                    desc = np.zeros((0, 128), dtype="uint8")
                descriptors.append(desc)
        else:
            self.logger.error(
                "Feature extractor: "
                + self.feature_extractor
                + " is not available, please choose bewteen SIFT, HOG and flatten"
            )
        if train:
            self.train_descriptors = copy.deepcopy(descriptors)
            if len(self.train_labels) == 0:
                self.train_labels = labels
        else:
            self.test_descriptors = copy.deepcopy(descriptors)
            if len(self.test_labels) == 0:
                self.test_labels = labels
        self.logger.info("Features extracted")
        if self.feature_extractor == "SIFT":
            self.histogram(descriptors, train=train)
        return descriptors

    def train_k_means(self, descriptors):
        """
        Normalize the data

        Parameters
        ----------
        data : list
            List of features

        Returns
        -------
        list
            List of normalized features
        """
        descriptors = np.vstack(descriptors)
        descriptors = descriptors.astype(np.float32)
        # Center descriptors
        mean = np.mean(descriptors)
        descriptors = np.apply_along_axis(lambda x: x - mean, axis=0, arr=descriptors)
        # PCA
        train_cov = np.dot(descriptors.T, descriptors)
        eigvals, eigvecs = np.linalg.eig(train_cov)
        perm = eigvals.argsort()
        pca_transform = eigvecs[:, perm[64:128]]
        descriptors = np.dot(descriptors, pca_transform)
        # Kmeans
        kmeans = KMeans(n_clusters=50)
        kmeans.fit(descriptors)
        # Save
        self.kmeans = kmeans
        self.pca_transform = pca_transform
        self.mean = mean

    def histogram(self, descriptors, train):
        """
        Create histogram of the data \n
        Automatically called by the extract method

        Parameters
        ----------
        data : list
            List of features

        Returns
        -------
        list
            List of histograms
        """
        if train:
            self.train_k_means(descriptors)
        elif self.kmeans is None and train is False:
            self.logger.error("Please train the kmeans first")
            return 0.4716
        self.logger.info("Kmeans fitted")
        image_descriptors = np.zeros((len(descriptors), self.kmeans.n_clusters), dtype=np.float32)
        for ii, desc in enumerate(descriptors):
            if desc.shape[0] == 0:
                self.logger.debug("WARNING: zero descriptor for %s" % (ii))
                continue
            # Convert to float32
            desc = desc.astype(np.float32)
            # Center descriptors
            desc = np.apply_along_axis(lambda x: x - self.mean, axis=0, arr=desc)
            desc = np.dot(desc, self.pca_transform)
            # Predict
            clabels = self.kmeans.predict(desc)
            # Histogram
            descr_hist = np.histogram(clabels, bins=self.kmeans.n_clusters)[0] / len(clabels)
            # Save
            image_descriptors[ii] = descr_hist
        self.logger.info("Histogram computed")
        if train:
            self.train_descriptors = copy.deepcopy(image_descriptors)
        else:
            self.test_descriptors = copy.deepcopy(image_descriptors)
        return image_descriptors

    def set_model(self, model):
        """
        Set a model to the classifier

        Parameters
        ----------
        model : sklearn model
            Initialized model to use
        """
        self.model = model

    def set_model_with_params(self, model, params):
        """
        Set a model to the classifier with parameters

        Parameters
        ----------
        model : sklearn model
            Model to use
        params : dict
            Parameters to set
        """
        self.model = model(**params)

    def fit_classifier(self):
        """
        Fit the classifier
        """
        if len(self.train_descriptors) == 0:
            self.logger.error("No train descriptors found, please extract features first")
            return
        if self.model is None:
            self.logger.error("No model found, please provide a model from sklearn")
            return
        self.logger.info("Fitting data on classifier, this may take a while...")
        self.model.fit(self.train_descriptors, self.train_labels)
        self.logger.info("Data fitted")

    def predict(self):
        """
        Predict the class of the data

        Parameters
        ----------
        data : list
            List of images

        Returns
        -------
        list
            List of predictions
        """
        if self.model is None:
            self.logger.error("No model found, please provide a model from sklearn")
            return
        if len(self.test_descriptors) == 0:
            self.logger.error("No test descriptors found, please extract features first")
            return
        return self.model.predict(self.test_descriptors)

    def evaluate(self):
        """
        Evaluate the model

        Parameters
        ----------
        data : list
            List of images

        Returns
        -------
        float
            Accuracy
        """
        if self.model is None:
            self.logger.error("No model found, please provide a model from sklearn")
            return
        if len(self.test_descriptors) == 0:
            self.logger.error("No test data found, please extract features first")
            return
        y_pred = self.predict()
        return accuracy_score(self.test_labels, y_pred)

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

    def do_all(self, path_to_train, path_to_test):
        """
        Do all the steps\n
        Initialize model before calling this method

        Parameters
        ----------
        path_to_train : str
            Path to the train data
        path_to_test : str
            Path to the test data
        """
        train_data, train_labels = self.load_data(path_to_train)
        test_data, test_labels = self.load_data(path_to_test)
        self.extract(train_data, train_labels, train=True)
        self.fit_classifier()
        self.extract(test_data, test_labels, train=False)
        accuracy = self.evaluate()
        self.save_model("svm.pkl")
        self.logger.info(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clf = sklearn.svm.SVC()
    model = Cifar10Classifier(feature_extractor="flatten", model=clf)
    model.do_all("data/external/data_batch_1", "data/external/test_batch")
