#Name: Gungun
#Course: CS-470-01
#Final Term Project
#Description: Fake News Detection
#Due Date: 21 April 2024

# Importing the Dependencies
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# Downloading NLTK resources
nltk.download('stopwords')

# Printing the stopwords in English
print("Stopwords in English:")
print(stopwords.words('english'))

# Loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv('train.csv')
print("Shape of the dataset:")
print(news_dataset.shape)

# Print the first 5 rows of the dataframe
print("First 5 rows of the DataFrame:")
print(news_dataset.head())

# Counting the number of missing values in the dataset
print("Number of missing values in each column:")
print(news_dataset.isnull().sum())

# Replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# Merging the author name and news title into a single column
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
print("First 5 rows of the merged content:")
print(news_dataset['content'].head())

# Separating the data & label
X = news_dataset['content']
Y = news_dataset['label']

# Stemming process
port_stem = PorterStemmer()

def stemming(content):
    # Remove all non-alphabet characters using regular expression
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert characters to lower case to maintain consistency
    stemmed_content = stemmed_content.lower()
    # Split the content into individual words (tokens)
    stemmed_content = stemmed_content.split()
    # Apply stemming to each word, removing stopwords along the way
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    # Join the list of words back into a single string separated by space
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
print("Content after stemming:")
print(news_dataset['content'].head())

# Vectorizing the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news_dataset['content'].values)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluating the Model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data:")
print(training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of the test data:")
print(test_data_accuracy)

# Display the model's coefficients and intercept
print("Model coefficients:")
print(model.coef_)
print("Model intercept:")
print(model.intercept_)

# Plotting the confusion matrix
cm = confusion_matrix(Y_test, X_test_prediction)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Calculating ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:,1])

# Plotting ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Filter out the true news from the dataset
true_news = news_dataset[Y == 0]

# Save the true news to a CSV file
true_news.to_csv('true_news.csv', index=False)

print("True news articles have been saved to same directory as 'true_news.csv'")
