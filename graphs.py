import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('spam_assassin.csv')

class_counts = dataset['target'].value_counts()
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=dataset)
plt.title('Dystrybucja klas e-maili')
plt.xlabel('Klasa')
plt.ylabel('Liczność')
plt.show()

def avg_length(word_list):
    total_length = sum(len(word) for word in word_list)
    average_length = total_length / len(word_list)
    return average_length

# Calculate average word length and total length of emails
dataset['word_length'] = dataset['text'].apply(lambda x: avg_length(x.split()))
dataset['total_length'] = dataset['text'].apply(lambda x: len(x.split()))

# Plot the box plots of average word length and total length
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='word_length', data=dataset)
plt.title('Średnia długość słowa w klasie')
plt.xlabel('Klasa')
plt.ylabel('Średnia długość słowa')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='total_length', data=dataset)
plt.title('Całkowita długość wiadomości(liczba słów) w klasie')
plt.xlabel('Klasa')
plt.ylabel('Całkowita długość')
plt.show()