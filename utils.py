import matplotlib
matplotlib.use('Agg')  # ✅ Force non-GUI backend

from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_digits_data():
    digits = datasets.load_digits()
    return digits

def train_svm_with_hyperparams(digits):
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.3, random_state=42
    )

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1]
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Score: {grid_search.score(X_test, y_test)}")

    return grid_search.best_estimator_

def plot_prediction(digits, clf):
    predicted = clf.predict([digits.data[-1]])
    print(f"Prediction: {predicted}")
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.savefig('output.png')
    print("Plot saved as output.png")