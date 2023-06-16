import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie danych z pliku CSV
data = pd.read_csv("C:/Users/acer/Desktop/spam_assassin.csv")

# Podział danych na cechy (X) i etykiety (y)
X = data["text"]
y = data["target"]

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przygotowanie cech za pomocą wektoryzacji TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Inicjalizacja i dopasowanie modelu Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = nb_model.predict(X_test)

# Obliczenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu: {:.2f}%".format(accuracy * 100))

# Wygenerowanie raportu klasyfikacji
classification_rep = classification_report(y_test, y_pred)
print("Raport klasyfikacji:")
print(classification_rep)
