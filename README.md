# Sentiment Analysis README

## Overview

This project involves building a sentiment analysis model using a dataset of tweets. The goal is to classify tweets as positive or negative based on their content. The model is built using Python and various machine learning libraries, including Scikit-learn and NLTK.

## Dataset

The dataset used is the Sentiment140 dataset from Kaggle, which contains 1.6 million tweets labeled as positive or negative.

### Source
- [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Dependencies

Ensure you have the following dependencies installed:

```sh
pip install kaggle
pip install scikit_learn
pip install nltk
```

## Steps

### 1. Kaggle API Configuration

- Place your `kaggle.json` file in the appropriate directory.

```sh
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Downloading the Dataset

Use the Kaggle API to download the Sentiment140 dataset.

```sh
kaggle datasets download -d kazanova/sentiment140
```

### 3. Extracting the Dataset

Extract the downloaded ZIP file.

```python
from zipfile import ZipFile

dataset = 'sentiment140.zip'
with ZipFile(dataset, 'r') as data:
    data.extractall()
    print("Dataset extracted")
```

### 4. Importing Libraries

Import the necessary libraries for data processing, machine learning, and natural language processing.

```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')
```

### 5. Data Loading and Preprocessing

Load the dataset and preprocess it by naming the columns, handling missing values, and stemming the text data.

```python
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', names=column_names, encoding='ISO-8859-1')
twitter_data.replace({'target': {4: 1}}, inplace=True)

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)
```

### 6. Data Splitting

Split the data into training and testing sets.

```python
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### 7. Feature Extraction

Convert the textual data to numerical data using TF-IDF Vectorizer.

```python
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
```

### 8. Model Training

Train a Logistic Regression model.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

### 9. Model Evaluation

Evaluate the model's accuracy on the training and testing data.

```python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of the training data: ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of the test data: ', test_data_accuracy)
```

### 10. Model Saving

Save the trained model for future use.

```python
import pickle

filename = "Trained_model.sav"
pickle.dump(model, open(filename, 'wb'))
```

### 11. Using the Saved Model

Load the saved model and make predictions.

```python
loaded_model = pickle.load(open('Trained_model.sav', 'rb'))

X_new = X_test[200]
print(Y_test[200])
prediction = model.predict(X_new)
print("Positive Tweet" if prediction[0] == 1 else "Negative Tweet")

X_new = X_test[3]
print(Y_test[3])
prediction = model.predict(X_new)
print("Positive Tweet" if prediction[0] == 1 else "Negative Tweet")
```

## Results

The model achieves an accuracy of approximately 77.8% on the test data.

## Acknowledgements

- Dataset: [Sentiment140 dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Libraries: Scikit-learn, NLTK, Pandas, NumPy

