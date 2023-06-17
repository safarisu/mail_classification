import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def avg_length(word_list):
    total_length = sum(len(word) for word in word_list)
    average_length = total_length / len(word_list)
    return average_length

dataset = pd.read_csv('spam_assassin.csv')

# Calculate length of each text string
dataset['text_length'] = dataset['text'].apply(lambda x: len(x.replace(" ", "")))
dataset['word_length'] = dataset['text'].apply(lambda x: avg_length(x.split()))
dataset['word_number'] = dataset['text'].apply(lambda x: len(x.split()))

# Calculate basic statistics
text_stats = dataset['text_length'].describe()

# Additional valuable insights
num_unique_texts = dataset['text'].nunique()
most_common_text = dataset['text'].value_counts().idxmax()

# Print basic statistics
print("Text length statistics for 'text' column:")
print(text_stats)
print("\n")

# Print additional valuable insights
print("Additional Insights for 'text' column:")
print("Number of unique texts:", num_unique_texts)
print("Most common text:", most_common_text)

# Calculate length of each text string
dataset['text_length'] = dataset['text'].apply(len)

# Calculate basic statistics for class 0
text_stats_class_0 = dataset[dataset['target'] == 0]['text_length'].describe()

# Calculate basic statistics for class 1
text_stats_class_1 = dataset[dataset['target'] == 1]['text_length'].describe()

# Additional valuable insights for class 0
num_unique_texts_class_0 = dataset[dataset['target'] == 0]['text'].nunique()
most_common_text_class_0 = dataset[dataset['target'] == 0]['text'].value_counts().idxmax()

# Additional valuable insights for class 1
num_unique_texts_class_1 = dataset[dataset['target'] == 1]['text'].nunique()
most_common_text_class_1 = dataset[dataset['target'] == 1]['text'].value_counts().idxmax()

# Print basic statistics for class 0
print("Text length statistics for 'text' column - Class 0:")
print(text_stats_class_0)
print("\n")

# Print basic statistics for class 1
print("Text length statistics for 'text' column - Class 1:")
print(text_stats_class_1)
print("\n")

# Print additional valuable insights for class 0
print("Additional Insights for 'text' column - Class 0:")
print("Number of unique texts:", num_unique_texts_class_0)
print("Most common text:", most_common_text_class_0)
print("\n")

# Print additional valuable insights for class 1
print("Additional Insights for 'text' column - Class 1:")
print("Number of unique texts:", num_unique_texts_class_1)
print("Most common text:", most_common_text_class_1)

# Calculate basic statistics for word length
word_length_stats = dataset['word_length'].describe()
print("Average word length statistics for 'text' column")
print(word_length_stats)
print("\n")

word_length_stats_class_0 = dataset[dataset['target'] == 0]['word_length'].describe()
print("Average word length statistics for 'text' column - Class 0")
print(word_length_stats_class_0)
print("\n")

word_length_stats_class_1 = dataset[dataset['target'] == 1]['word_length'].describe()
print("Word number statistics for 'text' column - Class 1")
print(word_length_stats_class_1)
print("\n")

# Calculate basic statistics for word number
word_length_stats = dataset['word_number'].describe()
print("Word number statistics for 'text' column")
print(word_length_stats)
print("\n")

word_length_stats_class_0 = dataset[dataset['target'] == 0]['word_number'].describe()
print("Word number statistics for 'text' column - Class 0")
print(word_length_stats_class_0)
print("\n")

word_length_stats_class_1 = dataset[dataset['target'] == 1]['word_number'].describe()
print("Word number statistics for 'text' column - Class 1")
print(word_length_stats_class_1)
print("\n")