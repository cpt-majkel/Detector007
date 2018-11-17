# FILENAME: create_dataset.py
# AUTHORS: Jalocha, Noga, Piekarski, Tancula

import cv2
from os import listdir
from os.path import isfile, join


class CreateDataset:
    """
    Note:
        base dir should contain 'test' and 'train' subfolders
        image naming convention: initials followed by "_", rest part is insignificant,
                                example  DC_001.jpg (when it is Daniel Craig )
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.train_dataset = []
        self.test_data = []
        self.initials2name = {
            "SC": "Sean Connery",
            "PB": "Pierce Brosnan",
            "DC": "Daniel Craig",
            "RM": "Roger Moore",
            "GL": "George Lazenby",
            "TD": "Timothy Dalton"
        }

    def read_test_data(self):
        path = self.base_dir + "/test"
        try:
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for bond in files:
                tmp = cv2.imread(path + "/" + bond)
                initials = bond.split("_")[0]
                self.test_data.append((tmp, self.initials2name[str(initials)]))
        except Exception as e:
            print("Following error occured while reading test data: ", e)

    def read_training_data(self):
        path = self.base_dir + "/train"
        try:
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for bond in files:
                tmp = cv2.imread(path + "/" + bond)
                initials = bond.split("_")[0]
                self.train_dataset.append((tmp, self.initials2name[str(initials)]))
        except Exception as e:
            print("Following error occured while reading training data: ", e)

    def read_data(self):
        self.read_test_data()
        self.read_training_data()
