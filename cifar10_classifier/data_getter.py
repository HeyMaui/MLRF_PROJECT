import pickle
import numpy as np
import copy
import logging

from cifar10_classifier.config import Config


class DataGetter:
    """
    Class to get the data from the CIFAR-10 dataset
    """

    def __init__(self, config: Config):
        self.conf = config
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def load_external_data(self):
        """
        Load the data from the path
        """
        data = []
        labels = []
        for i in range(1, 6):
            path = self.conf.DATA_DIR + f"/external/data_batch_{i}"
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
        self.train_data = copy.deepcopy(data)
        self.train_labels = copy.deepcopy(labels)
        data = []
        labels = []
        path = self.conf.DATA_DIR + f"/external/test_batch"
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
        self.test_data = copy.deepcopy(data)
        self.test_labels = copy.deepcopy(labels)
        self.logger.info("Data loaded")

    def save_data(self):
        """
        Save the data to a file
        """
        data = {
            "train_data": self.train_data,
            "train_labels": self.train_labels,
            "test_data": self.test_data,
            "test_labels": self.test_labels,
        }
        self.logger.info(f"Saving data to: {self.conf.DATA_DIR}/raw/all_data.pkl")
        with open(f"{self.conf.DATA_DIR}/raw/all_data.pkl", "wb+") as f:
            pickle.dump(data, f)

    def load_raw_data(self):
        """
        Load the data
        """
        self.logger.info(f"Loading data from: {self.conf.DATA_DIR}/raw/all_data.pkl")
        with open(f"{self.conf.DATA_DIR}/raw/all_data.pkl", "rb") as f:
            data = pickle.load(f)
        self.train_data = data["train_data"]
        self.train_labels = data["train_labels"]
        self.test_data = data["test_data"]
        self.test_labels = data["test_labels"]
        self.logger.info("Data loaded")
