#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on a sunny day...

@Student name : Naphatsakorn Khotsombat
@Student ID: R00146658
@Cohort: SD3
"""

from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("weatherAUS.csv", encoding='utf8')


def Task1():
    avg_rainfall = df.groupby('Location')['Rainfall'].mean().sort_values(ascending=False)
    avg_rainfall = avg_rainfall.reset_index()
    avg_rainfall.columns = ["Location", "Rainfall"]
    print(avg_rainfall)


def Task2():
    max_temp = df.groupby('Location')['MaxTemp'].max().sort_values(ascending=False)
    max_temp = max_temp.reset_index()
    max_temp.columns = ["Location", "MaxTemp"]
    print(max_temp)

    x = max_temp["Location"]
    y = max_temp["MaxTemp"]

    plt.figure()
    plt.plot(x, y)

    plt.xlim(-0.2, 5.2)
    plt.ylim(46.6, 48.1)

    plt.xlabel("Location")
    plt.ylabel("MaxTemp")
    plt.title("Location MaxTemp > 46.6")
    plt.show()


def Task3():
    dataset1 = df[["WindSpeed9am", "Humidity9am", "Pressure9am", "RainTomorrow"]].copy()

    dataset1 = dataset1[["WindSpeed9am", "Humidity9am", "Pressure9am", "RainTomorrow"]].dropna()

    allRainTomorrow = np.unique(dataset1['RainTomorrow'].astype(str))

    dict1 = {}
    c = 1
    for ac in allRainTomorrow:
        dict1[ac] = c
        c = c + 1

    dataset1["RainTomorrow"] = dataset1["RainTomorrow"].map(dict1)

    X_dataset1 = (dataset1[["WindSpeed9am", "Humidity9am", "Pressure9am"]])

    y_dataset1 = dataset1[["RainTomorrow"]]

    X_dataset1_train, X_dataset1_test, y_dataset1_train, y_dataset1_test = train_test_split(X_dataset1, y_dataset1,
                                                                                            test_size=0.67,
                                                                                            random_state=42)

    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X_dataset1_train, y_dataset1_train)

    print("Data Set 1 Training: ", tree_clf.score(X_dataset1_train, y_dataset1_train))
    print("Data Set 1 Testing: ", tree_clf.score(X_dataset1_test, y_dataset1_test))

    dataset2 = df[["WindSpeed3pm", "Humidity3pm", "Pressure3pm", "RainTomorrow"]].copy()

    dataset2 = dataset2[["WindSpeed3pm", "Humidity3pm", "Pressure3pm", "RainTomorrow"]].dropna()

    dataset2["RainTomorrow"] = dataset2["RainTomorrow"].map(dict1)

    X_dataset2 = (dataset2[["WindSpeed3pm", "Humidity3pm", "Pressure3pm"]])

    y_dataset2 = dataset2[["RainTomorrow"]]

    X_dataset2_train, X_dataset2_test, y_dataset2_train, y_dataset2_test = train_test_split(X_dataset2, y_dataset2,
                                                                                            test_size=0.33,
                                                                                            random_state=42)

    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X_dataset2_train, y_dataset2_train)

    print("Data Set 2 Training: ", tree_clf.score(X_dataset2_train, y_dataset2_train))
    print("Data Set 2 Testing: ", tree_clf.score(X_dataset2_test, y_dataset2_test))

    """
    Data set 1 training has a better accuracy than data set 2. But data set 2 has a better accuracy than 1.
    Data set 1 is better because it has more accuracy on the training one as it has more data on the training
    than in testing. 
    """

def Task4():
    dataset = df[["MaxTemp", "MinTemp", "WindGustSpeed", "Rainfall", "RainTomorrow"]].copy()

    dataset = dataset[["MaxTemp", "MinTemp", "WindGustSpeed", "Rainfall", "RainTomorrow"]].dropna()

    allRainTomorrow = np.unique(dataset['RainTomorrow'].astype(str))

    dict1 = {}
    c = 1
    for ac in allRainTomorrow:
        dict1[ac] = c
        c = c + 1

    dataset["RainTomorrow"] = dataset["RainTomorrow"].map(dict1)

    y = dataset[["RainTomorrow"]]

    for name in dataset[["MaxTemp", "MinTemp", "WindGustSpeed", "Rainfall"]]:
        X = (dataset[[name]])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = KNeighborsClassifier()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(f"Accuracy of rain fall tomorrow from {name} ", accuracy_score(y_test, y_pred))

    """
    Rainfall today has the most accuracy for predicting RainTomorrow because of the test that was done to each on
    element.
    """


def Task5():
    dataset = df[["WindGustDir", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]]

    dataset = dataset[["WindGustDir", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]].dropna()

    allWindGustDir = np.unique(dataset["WindGustDir"]).astype(str)

    dict1 = {}
    c = 1
    for ac in allWindGustDir:
        dict1[ac] = c
        c = c + 1

    dataset["WindGustDir"] = dataset["WindGustDir"].map(dict1)

    scalingObj = preprocessing.MinMaxScaler()
    newDataset = scalingObj.fit_transform(dataset)

    index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cost = []

    for i in range(10):
        kmeans = KMeans(n_clusters=i + 2).fit(newDataset)
        cost.append(kmeans.inertia_)
        print(kmeans.inertia_)

    plt.figure()
    plt.plot(index, cost)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within Clusters Sum of Squares")
    plt.title("Elbow Method")
    plt.show()


Task5()
