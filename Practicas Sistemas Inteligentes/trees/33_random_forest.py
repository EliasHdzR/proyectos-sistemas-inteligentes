from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn import datasets
import pandas as pd

dataset = pd.read_csv('../resources/agaricus-lepiota.csv', header=None)
dataset = dataset.sample(frac=1)
dataset.columns = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                   'gill-spacing',
                   'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                   'stalk-surface-below-ring', 'stalk-color-above-ring',
                   'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
                   'population',
                   'habitat']

#Encode the feature values which are strings to integers
for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])


X = dataset.drop(['target'],axis=1)
Y = dataset['target']

iris = datasets.load_iris()
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
X, Y = iris.data, iris.target

#Instantiate the model with 100 trees and entropy as splitting criteria
Random_Forest_model = RandomForestClassifier(n_estimators=100,criterion="entropy")


#Cross validation
accuracy = cross_validate(Random_Forest_model,X,Y,cv=10)['test_score']
print('The accuracy is: ',sum(accuracy)/len(accuracy)*100,'%')