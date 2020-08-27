from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
import pickle
import matplotlib.pyplot as plt


df = pd.read_csv('ionosphere.data', delimiter=',')
df.columns = [i for i in range(35)]
print(df.head)

# df.plot(subplots=True, layout=(18, 2))
# plt.show()
print(df.dtypes)

X = df.iloc[:, 0:34]
Y = df.iloc[:, 34]

seed = 5
test_size = 0.3

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, Y, test_size=test_size, random_state=seed)

mms = MinMaxScaler()
mms.fit(Xtrain)
Xtrain = pd.DataFrame(mms.transform(
    Xtrain), index=Xtrain.index, columns=Xtrain.columns)
Xtest = pd.DataFrame(mms.transform(
    Xtest), index=Xtest.index, columns=Xtest.columns)

lb = LabelBinarizer()
Ytrain = pd.DataFrame(lb.fit_transform(Ytrain))
Ytest = pd.DataFrame(lb.fit_transform(Ytest))

nb = MultinomialNB(alpha=0.1)
nb.fit(Xtrain, Ytrain)
ypred = nb.predict(Xtest)
accuracy = accuracy_score(Ytest, ypred)
print(f'accuracy score = {accuracy}%')


model = LogisticRegression()
model.fit(Xtrain, Ytrain.values.ravel())
ypred = nb.predict(Xtest)
accuracy = accuracy_score(Ytest, ypred)
print(f'accuracy score = {accuracy}%')
