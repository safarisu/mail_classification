import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# Wczytanie danych z pliku CSV
data = pd.read_csv("spam_assassin.csv")

# Podział danych na cechy (X) i etykiety (y)
X = data["text"]
y = data["target"]

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przygotowanie cech za pomocą wektoryzacji TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Inicjalizacja i dopasowanie różnych modeli
model1 = LogisticRegression()
model1.fit(X_train, y_train)

model2 = MultinomialNB()
model2.fit(X_train, y_train)

model3 = SVC(probability=True)
model3.fit(X_train, y_train)

# Stworzenie hybrydowego modelu łączącego wyniki różnych modeli
hybrid_model = VotingClassifier(estimators=[('lr', model1), ('nb', model2), ('svc', model3)], voting='soft')
hybrid_model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = hybrid_model.predict(X_test)

# Obliczenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu: {:.2f}%".format(accuracy * 100))

# Wygenerowanie raportu klasyfikacji
classification_rep = classification_report(y_test, y_pred)
print("Raport klasyfikacji:")
print(classification_rep)

# Generowanie macierzy pomyłek
cm = confusion_matrix(y_test, y_pred)

# Tworzenie heatmapy macierzy pomyłek
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Macierz Pomyłek')
plt.xlabel('Predykowana etykieta')
plt.ylabel('Rzeczywista etykieta')
plt.show()
