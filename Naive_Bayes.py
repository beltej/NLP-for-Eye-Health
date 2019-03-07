import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
#from sklearn import cross_validation
# from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import matplotlib.pyplot as plt


def model_accuracy_graph(modelNaive, X, y):
    values_of_score = []
    folds = []
    for i in range(3,11):

        answer = cross_val_score(modelNaive, X, y, cv=i, scoring='accuracy')
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

    # modelRandom = RandomForestClassifier(max_depth=3)
    modelNaive = GaussianNB()

    # modelRandom_crossValidation = cross_val_score(modelRandom, X, y, cv=5, scoring='accuracy')
    modelNaive_crossValidation = cross_val_score(modelNaive, X, y, cv=5, scoring='accuracy')

    model_accuracy_graph(modelNaive, X, y)

    print("\n Mean model accuracy of 5 folds:")
    # print(modelRandom_crossValidation.mean())
    print(modelNaive_crossValidation.mean())

    # modelRandom.fit(X, y)
    modelNaive.fit(X,y)

    # predictions = modelRandom.predict(test[predictor_vars])

    predictions = modelNaive.predict(test[predictor_vars])
    print ("\n Predictions:")
    print (predictions)

    print ("\n Accuracy comparing predictions with actual values:")
    print (metrics.accuracy_score(train.category, predictions))

    print ("\n Precision, recall, fscore:")
    precision, recall, F1, _ = list(precision_recall_fscore_support(train.category, predictions, beta=1.0, average="binary"))
    print "Precision: %f " % (precision,)
    print "Recall: %f " % (recall,)
    print "F1: %f " % (F1,)


def train_on_gold_std_test_on_10k():

    train = pd.read_csv('/Users/tejasvibelsare/Documents/Gold_Standard_Data.csv')
    test = pd.read_csv('/Users/tejasvibelsare/Documents/Gold_Standard_Data.csv')

    predictor_vars = ["polarity", "matching_number_of_keywords_allergic", "matching_number_of_keywords_infectious",
                      "length_of_tweet"]
    # predictor_vars = ["matching_number_of_keywords_allergic", "matching_number_of_keywords_infectious"]

    X, y = train[predictor_vars], train.category

    # modelRandom = RandomForestClassifier(max_depth=3)
    modelNaive = GaussianNB()

    # modelRandom_crossValidation = cross_val_score(modelRandom, X, y, cv=5, scoring='accuracy')
    modelNaive_crossValidation = cross_val_score(modelNaive, X, y, cv=5, scoring='accuracy')

    print("Mean model accuracy of 5 folds:")
    # print(modelRandom_crossValidation.mean())
    print(modelNaive_crossValidation.mean())

    # modelRandom.fit(X, y)
    modelNaive.fit(X,y)

    # predictions = modelRandom.predict(test[predictor_vars])

    predictions = modelNaive.predict(test[predictor_vars])
    print ("Predictions:")
    print (predictions)

    print ("Accuracy comparing predictions with actual values:")
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


