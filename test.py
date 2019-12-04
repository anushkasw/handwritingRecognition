"""Team: Bazinga"""
import sys
import pickle
import loaddata
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as skt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score

def predict():
    """ Load Test data. """
    input_data = loaddata.load_pkl(sys.argv[2])
    np.save('input_pickle_data.npy', input_data)
    test_data = np.load('input_pickle_data.npy', allow_pickle=True)


    """Network type"""
    if sys.argv[1] == 'AB':
        networkType = 'AB'
    else:
        networkType = 'All'


    def data_process(train_data):
        resized_data = []
        for item in train_data:
            resized_data.append(skt.resize(np.float32(item), (100, 100)))
        for i in range(len(resized_data)):
            resized_data[i] = resized_data[i].flatten()
        resized_data = np.asarray(resized_data)
        return resized_data

    """ Obtain all images of 'a' and 'b'. Create a dataset with those instances."""
    def data_split(resized_data, train_labels):
        set_X, set_Y = [], []
        for i in (np.unique(train_labels)):
            items = list(np.where(train_labels == i)[0])
            set_X.append(resized_data[items])
            set_Y.append(train_labels[items])

        if networkType == "AB":
            x = np.concatenate((np.asarray(set_X[0]), np.asarray(set_X[1])))
            y = np.concatenate((np.asarray(set_Y[0]), np.asarray(set_Y[1])))
        else:
            x = np.concatenate((np.asarray(set_X[0]), np.asarray(set_X[1]), np.asarray(set_X[2]), np.asarray(set_X[3]), np.asarray(set_X[4]), np.asarray(set_X[5]), np.asarray(set_X[6]), np.asarray(set_X[7])))
            y = np.concatenate((np.asarray(set_Y[0]), np.asarray(set_Y[1]), np.asarray(set_Y[2]), np.asarray(set_Y[3]), np.asarray(set_Y[4]), np.asarray(set_Y[5]), np.asarray(set_Y[6]), np.asarray(set_Y[7])))
        return x, y

    """Resize the test data"""
    test_resize_data = data_process(test_data)

    """ Load trained network"""
    networkFile = 'networkAB.pickle' if networkType == "AB" else "networkAll.pickle"
    with open(networkFile, 'rb') as handle:
        network = pickle.load(handle)
    load_lr_model = pickle.load(open(networkFile, 'rb'))

    """Forward the test data into network"""
    y_predicted = load_lr_model.predict(test_resize_data)


    """ Saving predicted values in an output file"""
    np.save(sys.argv[3], y_predicted)
    return y_predicted

if __name__ == "__main__":
    y = predict()
    print('Predicted values are: ', y)
