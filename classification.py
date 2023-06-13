import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Load the spam email dataset
dataset = pd.read_csv('spam_assassin.csv')
X = dataset['text']
y = dataset['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a CountVectorizer to convert email text into feature vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the training data
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Predict the labels for the test data
y_pred = classifier.predict(X_test_counts)

# Print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()

clf = RandomForestClassifier(random_state = 42, n_jobs=-1)
clf.fit(X_train_counts, y_train)
y_pred = clf.predict(X_test_counts)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()

k=3
KNN = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
KNN.fit(X_train_counts, y_train)
y_pred = KNN.predict(X_test_counts)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()