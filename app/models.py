import pickle

class Pickle_model():
    def get_pickle(self, path):
        return pickle.load(open(path, 'rb'))