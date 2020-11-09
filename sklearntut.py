# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Loading dataset
wine = pd.read_csv('/home/will/Development/Python/scikit-learn/winequality-red.csv')
# Getting some info about the data
# print(wine.head())
# print(wine.info())
# Checking to see if there's any null values in the data.
# If they're statistically insignifigant, we can remove them.
# print(wine.isnull().sum())

# preprocessing data
# creating 2 bins, 6.5 or above to be 'good'. 8 is the max for good

bins = (2, 6.5, 8)
group_names = ['bad', 'good']

# use pandas to cut the data into our two bins

wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
# print(wine['quality'].unique())
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
# print(wine['quality'].value_counts())
# sns.countplot(wine['quality'])
# plt.show()

# now seperate the data set as response variables and feature variables
# The variables we want to use to predict our response

X = wine.drop('quality', axis=1)

# The variable we are trying to predict

y = wine['quality']

# Train and test splitting of data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying Standard scaling to get optimized result
# This is important because one variable might be 100 and another might be 1, and we want
# to weigh each variable evenly.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train[:10])

# RandomForestClassifier Model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(pred_rfc[:20])
# Let's see how it performed
#print(classification_report(y_test, pred_rfc))
#print(confusion_matrix(y_test, pred_rfc))

# SVM Classifier
clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
# Lets see how it performed
# print(classification_report(y_test, pred_clf))
# print(confusion_matrix(y_test, pred_clf))

# Neural Network
# layers = number of variables sometimes
mlpc = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
#print(classification_report(y_test, pred_mlpc))
#print(confusion_matrix(y_test, pred_mlpc))

# Scoring the accuracy of the model
cm = accuracy_score(y_test, pred_rfc)

# Now, lets use our models to predict using new data

Xnew = [[7.3, 0.58, 0.00, 2.0, 0.065, 15.0, 21.0, 0.9946, 3.36, 0.47, 10.0]]
Xnew = sc.transform(Xnew)
ynew = rfc.predict(Xnew)
print("quality of xnew: "+str(ynew))
