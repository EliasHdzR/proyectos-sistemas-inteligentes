from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Test data
test_data = [
    ["Strawberry", "Red", "Smooth", "Sweet", "Desserts", "Fruit"],
    ["Celery", "Green", "Crisp", "Mild", "Salads", "Vegetable"],
    ["Pineapple", "Yellow", "Rough", "Sweet", "Snacks", "Fruit"],
    ["Spinach", "Green", "Tender", "Mild", "Salads", "Vegetable"],
    ["Blueberry", "Blue", "Smooth", "Sweet", "Baking", "Fruit"],
    ["Cucumber", "Green", "Crisp", "Mild", "Salads", "Vegetable"],
    ["Watermelon", "Red", "Juicy", "Sweet", "Snacks", "Fruit"],
    ["Carrot", "Orange", "Crunchy", "Sweet", "Salads", "Vegetable"],
    ["Grapes", "Purple", "Juicy", "Sweet", "Snacks", "Fruit"],
    ["Bell Pepper", "Red", "Crisp", "Mild", "Cooking", "Vegetable"],
    ["Kiwi", "Brown", "Fuzzy", "Tart", "Snacks", "Fruit"],
    ["Lettuce", "Green", "Tender", "Mild", "Salads", "Vegetable"],
    ["Mango", "Orange", "Smooth", "Sweet", "Desserts", "Fruit"],
    ["Potato", "Brown", "Starchy", "Mild", "Cooking", "Vegetable"],
    ["Apple", "Red", "Crunchy", "Sweet", "Snacks", "Fruit"],
    ["Onion", "White", "Firm", "Pungent", "Cooking", "Vegetable"],
    ["Orange", "Orange", "Smooth", "Sweet", "Snacks", "Fruit"],
    ["Garlic", "White", "Firm", "Pungent", "Cooking", "Vegetable"],
    ["Peach", "Orange", "Smooth", "Sweet", "Desserts", "Fruit"],
    ["Broccoli", "Green", "Tender", "Mild", "Cooking", "Vegetable"],
    ["Cherry", "Red", "Juicy", "Sweet", "Snacks", "Fruit"],
    ["Peas", "Green", "Soft", "Sweet", "Cooking", "Vegetable"],
    ["Pear", "Green", "Juicy", "Sweet", "Snacks", "Fruit"],
    ["Cabbage", "Green", "Crisp", "Mild", "Cooking", "Vegetable"],
    ["Grapefruit", "Pink", "Juicy", "Tart", "Snacks", "Fruit"],
    ["Asparagus", "Green", "Tender", "Mild", "Cooking", "Vegetable"]
]

# Separate features and labels
X_test = [row[:-1] for row in test_data]
y_test = [row[-1] for row in test_data]

# Encoding categorical features
label_encoders = [LabelEncoder() for _ in range(4)]  # One encoder for each categorical feature

# Fit and transform each categorical feature
X_encoded = []
for i, encoder in enumerate(label_encoders):
    feature_values = [row[i+1] for row in test_data]  # Extract values for the current feature
    encoded_feature = encoder.fit_transform(feature_values)
    X_encoded.append(encoded_feature)

# Transpose X_encoded to get features in rows
X_encoded = list(map(list, zip(*X_encoded)))

# Initialize and train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_encoded, y_test)

# Predict on the test data
y_pred = nb_classifier.predict(X_encoded)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

exit()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data
np.random.seed(0)

# Define the features: BMI and sports activity level
num_samples = 500

bmi = np.random.uniform(18.5, 35, num_samples)  # BMI values between 18.5 and 35
# 0, 1, 2 corresponding to 'low', 'moderate', 'high':
activity_level = np.random.choice([0, 1, 2], size=num_samples)  # Activity levels

# Define the target labels: 0 for low risk, 1 for high risk
# The risk increases if BMI is high (>= 30) and activity level is low
labels = ((bmi >= 30) & (activity_level == 0)).astype(int)
print(bmi)
print(activity_level)
print(labels)

data = np.column_stack((bmi, activity_level))
print(data.shape)



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
naive_bayes_classifier = GaussianNB()

# Train the classifier on the training data
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)



exit()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = [
    ["Strawberry", "Red", "Smooth", "Sweet", "Desserts", "Fruit"],
    ["Celery", "Green", "Crisp", "Mild", "Salads", "Vegetable"],
    ["Pineapple", "Yellow", "Rough", "Sweet", "Snacks", "Fruit"],
    ["Spinach", "Green", "Tender", "Mild", "Salads", "Vegetable"],
    ["Blueberry", "Blue", "Smooth", "Sweet", "Baking", "Fruit"],
    ["Cucumber", "Green", "Crisp", "Mild", "Salads", "Vegetable"],
    ["Watermelon", "Red", "Juicy", "Sweet", "Snacks", "Fruit"],
    ["Carrot", "Orange", "Crunchy", "Sweet", "Salads", "Vegetable"],
    ["Grapes", "Purple", "Juicy", "Sweet", "Snacks", "Fruit"],
    ["Bell Pepper", "Red", "Crisp", "Mild", "Cooking", "Vegetable"],
    ["Kiwi", "Brown", "Fuzzy", "Tart", "Snacks", "Fruit"],
    ["Lettuce", "Green", "Tender", "Mild", "Salads", "Vegetable"],
    ["Mango", "Orange", "Smooth", "Sweet", "Desserts", "Fruit"],
    ["Potato", "Brown", "Starchy", "Mild", "Cooking", "Vegetable"],
    ["Apple", "Red", "Crunchy", "Sweet", "Snacks", "Fruit"],
    ["Onion", "White", "Firm", "Pungent", "Cooking", "Vegetable"],
    ["Orange", "Orange", "Smooth", "Sweet", "Snacks", "Fruit"],
    ["Garlic", "White", "Firm", "Pungent", "Cooking", "Vegetable"],
    ["Peach", "Orange", "Smooth", "Sweet", "Desserts", "Fruit"],
    ["Broccoli", "Green", "Tender", "Mild", "Cooking", "Vegetable"],
    ["Cherry", "Red", "Juicy", "Sweet", "Snacks", "Fruit"],
    ["Peas", "Green", "Soft", "Sweet", "Cooking", "Vegetable"],
    ["Pear", "Green", "Juicy", "Sweet", "Snacks", "Fruit"],
    ["Cabbage", "Green", "Crisp", "Mild", "Cooking", "Vegetable"],
    ["Grapefruit", "Pink", "Juicy", "Tart", "Snacks", "Fruit"],
    ["Asparagus", "Green", "Tender", "Mild", "Cooking", "Vegetable"]
]

# Convert data to numpy array
data = np.array(data)

# Split data into features (X) and labels (y)
X = data[:, :-1]  # Features (all columns except the last one)
y = data[:, -1]   # Labels (last column)

# Encoding categorical features
label_encoders = [LabelEncoder() for _ in range(X.shape[1])]
X_encoded = np.zeros(X.shape)
for i, encoder in enumerate(label_encoders):
    X_encoded[:, i] = encoder.fit_transform(X[:, i])

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)
print(X_train[:10])

# Initialize and train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

exit()

# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = GaussianNB()

model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))