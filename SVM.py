from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


def model_accuracy_graph(modelSVM, X, y):
    values_of_score = []
    folds = []
    for i in range(3,11):

        answer = cross_val_score(modelSVM, X, y, cv=i, scoring='accuracy')
        values_of_score.append(answer.mean())
        folds.append(i)

    # print("Values of score:",values_of_score)
    # print("folds:",folds)
    plt.plot(folds,values_of_score)

    plt.xlabel("Number of folds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model accuracy graph for different folds")

    plt.show()


def train_and_test_on_gold_std():

    train = pd.read_csv('/Users/tejasvibelsare/Documents/Gold_Standard_Data.csv')
    test = pd.read_csv('/Users/tejasvibelsare/Documents/Gold_Standard_Data.csv')

    predictor_vars = ["polarity", "matching_number_of_keywords_allergic", "matching_number_of_keywords_infectious",
                      "length_of_tweet"]
    # predictor_vars = ["matching_number_of_keywords_allergic", "matching_number_of_keywords_infectious"]

    X, y = train[predictor_vars], train.category

    # modelSVM = svm.SVC()
    modelSVM = svm.SVC(kernel='linear')

    modelSVM_crossValidation = cross_val_score(modelSVM,X,y,cv=5,scoring='accuracy')

    model_accuracy_graph(modelSVM, X, y)

    print("\n Mean model accuracy of 5 folds:")
    print(modelSVM_crossValidation.mean())

    modelSVM.fit(X,y)

    # predictions = modelRandom.predict(test[predictor_vars])

    predictions = modelSVM.predict(test[predictor_vars])
    print ("\n Predictions:")
    print (predictions)

    print ("\n Accuracy comparing predictions with actual values:")
    print (metrics.accuracy_score(train.category, predictions))

    print ("\n Precision, recall, fscore, support:")
    print (precision_recall_fscore_support(train.category, predictions, beta=1.0, average="binary"))


def train_on_gold_std_test_on_10k():

    train = pd.read_csv('/Users/tejasvibelsare/Documents/Gold_Standard_Data.csv')
    test = pd.read_csv('/Users/tejasvibelsare/Documents/Gold_Standard_Data.csv')

    predictor_vars = ["polarity", "matching_number_of_keywords_allergic", "matching_number_of_keywords_infectious",
                      "length_of_tweet"]
    # predictor_vars = ["matching_number_of_keywords_allergic", "matching_number_of_keywords_infectious"]

    X, y = train[predictor_vars], train.category

    modelSVM = svm.SVC(kernel='linear')

    modelSVM_crossValidation = cross_val_score(modelSVM,X,y,cv=5,scoring='accuracy')



    # modelSVM_crossValidation = cross_val_score(X, y, cv=5, scoring='accuracy')

    print("\n Mean model accuracy of 5 folds:")
    print(modelSVM_crossValidation.mean())

    modelSVM.fit(X,y)

    # predictions = modelRandom.predict(test[predictor_vars])

    predictions = modelSVM.predict(test[predictor_vars])
    print ("\n Predictions:")
    print (predictions)

    print ("\n Accuracy comparing predictions with actual values:")
    print (metrics.accuracy_score(train.category, predictions))

    print ("\n Precision, recall, fscore:")
    precision, recall, F1, _ = list(precision_recall_fscore_support(train.category, predictions, beta=1.0, average="binary"))
    print "Precision: %f " % (precision,)
    print "Recall: %f " % (recall,)
    print "F1: %f " % (F1,)


def main():

    train_and_test_on_gold_std()

    # train_on_gold_std_test_on_10k()


if __name__ == "__main__":
    # calling main function
    main()


