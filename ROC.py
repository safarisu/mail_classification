import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

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

# Inicjalizacja i dopasowanie modelu Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Obliczenie prawdopodobieństwa predykcji dla etykiety pozytywnej (spam)
y_scores = gb_model.predict_proba(X_test)[:, 1]

# Wygenerowanie krzywej ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Wykres krzywej ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (obszar pod krzywą = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Odsetek fałszywie pozytywnych')
plt.ylabel('Odsetek prawdziwie pozytywnych')
plt.title('Krzywa ROC')
plt.legend(loc="lower right")
plt.show()
