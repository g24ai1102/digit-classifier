import matplotlib
matplotlib.use('Agg')  # ✅ Force non-GUI backend

from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

def load_digits_data():
    digits = datasets.load_digits()
    return digits

def train_svm_classifier(digits):
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(digits.data[:-1], digits.target[:-1])
    return clf

def plot_prediction(digits, clf):
    predicted = clf.predict([digits.data[-1]])
    print(f"Prediction: {predicted}")
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.savefig('output.png')
    print("Plot saved as output.png")