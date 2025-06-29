from utils import load_digits_data, train_svm_with_hyperparams, plot_prediction

def main():
    digits = load_digits_data()
    clf = train_svm_with_hyperparams(digits)
    plot_prediction(digits, clf)

if __name__ == "__main__":
    main()