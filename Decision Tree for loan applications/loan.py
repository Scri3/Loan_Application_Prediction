#Three lines to make our compiler able to draw:
import matplotlib
matplotlib.use('Agg')

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# In the beginning, we need to convert the csv file to a dataframe
DataFrame = pandas.read_csv("data.csv")


# All the data must be numerical, so we should assign a number for the fields that contain string value

Result = {'Approved': 1, 'Denied': 0}
DataFrame['loan_status'] = DataFrame['loan_status'].map(Result)


features = ['age', 'credit_score']

X = DataFrame[features]
y = DataFrame['loan_status']

DecisionTree = DecisionTreeClassifier()
DecisionTree = DecisionTree.fit(X, y)

tree.plot_tree(DecisionTree, feature_names=features)

plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")