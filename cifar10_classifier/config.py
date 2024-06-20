import os


class Config:
    def __init__(self, PROJECT_ROOT):
        self.PROJECT_ROOT = PROJECT_ROOT
        self.DATA_DIR = os.path.join(PROJECT_ROOT, "../data")
        self.MODELS_DIR = os.path.join(PROJECT_ROOT, "../models")
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
