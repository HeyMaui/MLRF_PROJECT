import logging
import cv2
import numpy as np
import copy
from skimage.feature import hog
from sklearn.cluster import KMeans


class FeatureExtractor:

    def __init__(self, feature_extractor="HOG"):
        self.feature_extractor = feature_extractor
        self.train_descriptors = []
        self.train_labels = []
        self.test_descriptors = []
        self.test_labels = []
        self.kmeans = None
        self.pca_transform = None
        self.mean = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

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
