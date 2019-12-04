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

"""Network type"""
networkType = 'AB' if sys.argv[1] == 'AB' else 'All'

""" Load training data"""
input_train_data = loaddata.load_pkl('train_data.pkl')
np.save('input_pickle_data.npy', input_train_data)
train_data = np.load('input_pickle_data.npy', allow_pickle=True)
train_labels = np.load('finalLabelsTrain.npy')


""" Flatten the image data into 1D array to be fed into the neural network."""
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

resized_data = data_process(train_data)
x, y = data_split(resized_data, train_labels)

"""Cross-Validation"""
kf = KFold(n_splits=5)
dnns_classifier = OneVsRestClassifier(MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 3), max_iter=2500,
                                                    early_stopping=True, momentum=0.05, activation='tanh', learning_rate_init=0.1))

for train_indices, test_indices in kf.split(x):
    dnns_classifier.fit(x[train_indices], y[train_indices])

"""Save the network model"""
networkFile = 'networkAB.pickle' if networkType == "AB" else "networkAll.pickle"
with open(networkFile, 'wb') as handle:
    pickle.dump(dnns_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

