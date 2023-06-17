import collections

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

dataset = pd.read_csv('spam_assassin.csv')

# Calculate length of each text string
dataset['text_length'] = dataset['text'].apply(lambda x: len(x.replace(" ", "")))

class_counts = dataset['target'].value_counts()
classes = dataset['target'].unique()
fig = plt.figure(figsize=(8, 6))
ax = sns.countplot(x='target', data=dataset, palette=sns.color_palette("bright"))
ax.bar_label(container=ax.containers[0], labels=class_counts)

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

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first box plot - average word length
sns.boxplot(ax=axes[0], x='target', y='word_length', data=dataset, palette="bright")
axes[0].set_title('Średnia długość słowa w klasie')
axes[0].set_xlabel('Klasa')
axes[0].set_ylabel('Średnia długość słowa')

# Plot the second box plot - total length
sns.boxplot(ax=axes[1], x='target', y='total_length', data=dataset, palette="bright")
axes[1].set_title('Całkowita długość wiadomości (liczba słów) w klasie')
axes[1].set_xlabel('Klasa')
axes[1].set_ylabel('Całkowita długość')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Removing outliers
z_scores_word = np.abs((dataset['word_length'] - dataset['word_length'].mean()) / dataset['word_length'].std())
z_scores_total = np.abs((dataset['total_length'] - dataset['total_length'].mean()) / dataset['total_length'].std())

# Remove outliers above 3 standard deviations for word length
dataset_filtered = dataset[z_scores_word <= 3]

# Remove outliers above 3 standard deviations for total length
dataset_filtered = dataset_filtered[z_scores_total <= 3]

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first box plot - average word length
sns.boxplot(ax=axes[0], x='target', y='word_length', data=dataset_filtered, palette="bright")
axes[0].set_title('Średnia długość słowa w klasie')
axes[0].set_xlabel('Klasa')
axes[0].set_ylabel('Średnia długość słowa')

# Plot the second box plot - total length
sns.boxplot(ax=axes[1], x='target', y='total_length', data=dataset_filtered, palette="bright")
axes[1].set_title('Całkowita długość wiadomości (liczba słów) w klasie')
axes[1].set_xlabel('Klasa')
axes[1].set_ylabel('Całkowita długość')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Removing text length outliers

# Removing outliers
z_scores_text = np.abs((dataset['text_length'] - dataset['text_length'].mean()) / dataset['text_length'].std())

# Remove outliers above 3 standard deviations for text length
dataset_filtered = dataset[z_scores_text <= 3]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.boxplot(ax=axes[0], x='target', y='text_length', data=dataset, palette="bright")
axes[0].set_title('Całkowita długość tekstu(bez spacji)')
axes[0].set_xlabel('Klasa')
axes[0].set_ylabel('Liczba znaków')

sns.boxplot(ax=axes[1], x='target', y='text_length', data=dataset_filtered, palette="bright")
axes[1].set_title('Całkowita długość tekstu(bez spacji) - po usunięciu outlierów')
axes[1].set_xlabel('Klasa')
axes[1].set_ylabel('Liczba znaków')

plt.show()

def calculate_entropy(text):
    char_counts = collections.Counter(text)
    total_chars = len(text)
    char_probabilities = [count / total_chars for count in char_counts.values()]
    text_entropy = entropy(char_probabilities)
    return text_entropy

dataset['alphabetic_ratio'] = dataset['text'].apply(lambda x: sum(c.isalpha() for c in x.replace(" ", "")) / len(x.replace(" ", "")))

# Calculate entropy for the "text" column
dataset['text_entropy'] = dataset['text'].apply(calculate_entropy)

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Box plot of alphabetic ratio
sns.boxplot(ax=axes[0], x='target', y='alphabetic_ratio', data=dataset, palette="bright")
axes[0].set_title('Współczynnik znaków alfabetycznych we wiadomości')
axes[0].set_ylabel('Alphabetic Ratio')

# Box plot of entropy values
sns.boxplot(ax=axes[1], x='target', y='text_entropy', data=dataset, palette="bright")
axes[1].set_title('Entropia wiadomości')
axes[1].set_ylabel('Entropia')

# Adjust spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()

print(dataset.info())

# Select the relevant columns for correlation
columns_to_correlate = ['target', 'text_length', 'word_length', 'total_length', 'alphabetic_ratio', 'text_entropy']
correlation_data = dataset[columns_to_correlate]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix')
plt.show()