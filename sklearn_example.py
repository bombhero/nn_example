from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
import numpy as np
import pandas as pd


def stock_test():
    data = np.loadtxt(fname="e:/bomb/proj/StockDetection_py3v2_data/Stock_org/2019-09-07/rdata_new/000070_ready.csv"
                      , delimiter=",")
    x_data = data[:, 0:480]
    y_data = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("score: {}".format(score))

    dot_tree = tree.export_graphviz(clf)
    graph = graphviz.Source(dot_tree)
    # graph.view()

    feature_names = []
    for i in range(480):
        feature_names.append("f%03d" % i)
    df = pd.DataFrame([*zip(feature_names, clf.feature_importances_)])
    df.to_csv(path_or_buf="feature_important.csv", sep=",")


def main():
    dataset = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("score: {}".format(score))

    dot_tree = tree.export_graphviz(clf,
                                    feature_names=dataset.feature_names,
                                    class_names=dataset.target_names)
    graph = graphviz.Source(dot_tree)
    graph.view()


if __name__ == "__main__":
    stock_test()
