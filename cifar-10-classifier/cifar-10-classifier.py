import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class Cifar10Classifier:
    def __init__(self, feature_extractor="SIFT", classifier="", model=None):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.descriptors = []
        self.model = model

    def extract(self, data):
        for img in data:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if self.feature_extractor == "SIFT":
                fe = cv2.SIFT.create(nfeatures=100)
            elif self.feature_extractor == "SURF":
                fe = cv2.SURF.create(nfeatures=100)
            elif self.feature_extractor == "HOG":
                pass
            else:
                print(
                    "Feature extractor: "
                    + self.feature_extractor
                    + " is not available, please choose bewteen SIFT, ORB and ...."
                )
                return
            kpts = fe.detect(img_gray)
            kpts, desc = fe.compute(img_gray, kpts)
            self.descriptors.append(desc)
        return self.descriptors
