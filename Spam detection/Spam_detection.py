import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.simplefilter("ignore")

# NLTK and Sklearn libraries for text processing and model building
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load the dataset
df= pd.read_csv('C:\\Users\\YASHIKA NEGI\\Documents\\Yashika\\VS code\\vs code python\\OIBSIP-Data Science\\Task 3\\spam.csv', encoding='latin1')

# Renaming columns for better understanding
df.rename(columns={"v1": "Label", "v2": "messages"}, inplace=True)

# Dropping unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Checking and removing duplicates
df = df.drop_duplicates()

# Bar plot showing distribution of labels (Spam vs. Ham)
plt.figure(figsize=(6, 4))
sns.countplot(x='Label', data=df, palette='Set2')
plt.title('Distribution of Spam vs. Ham Messages')
plt.show()

# Word cloud for spam and ham messages
spam_words = ' '.join(list(df[df['Label'] == 'spam']['messages']))
ham_words = ' '.join(list(df[df['Label'] == 'ham']['messages']))

wordcloud_spam = WordCloud(width=600, height=400, background_color='white').generate(spam_words)
wordcloud_ham = WordCloud(width=600, height=400, background_color='white').generate(ham_words)

# Plot the word clouds
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.title('Spam Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.title('Ham Word Cloud')
plt.axis('off')
plt.show()

# Text preprocessing
ps = PorterStemmer()
corpus = []
for i in range(len(df)):
    rp = re.sub("[^a-zA-Z]", " ", df["messages"].iloc[i])
    rp = rp.lower()
    rp = rp.split()
    rp = [ps.stem(word) for word in rp if word not in set(stopwords.words("english"))]
    rp = " ".join(rp)
    corpus.append(rp)

# Vectorizing text data
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(df["Label"], drop_first=True)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47)

# Model training with Naive Bayes
model = MultinomialNB()
model.fit(x_train, y_train)

# Predictions for Naive Bayes
ŷ_train_nb = model.predict(x_train)
ŷ_test_nb = model.predict(x_test)

print("Naive Bayes train accuracy:", accuracy_score(y_train, ŷ_train_nb))
print("Naive Bayes test accuracy:", accuracy_score(y_test, ŷ_test_nb))

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, ŷ_test_nb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Logistic Regression Model
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Predictions for Logistic Regression
ŷ_train_lr = lr.predict(x_train)
ŷ_test_lr = lr.predict(x_test)

print("Logistic Regression train accuracy:", accuracy_score(y_train, ŷ_train_lr))
print("Logistic Regression test accuracy:", accuracy_score(y_test, ŷ_test_lr))
print("Cross-validation score:", cross_val_score(lr, x_train, y_train, cv=5).mean())

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, ŷ_test_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap='Greens', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Comparing accuracy of Naive Bayes and Logistic Regression
plt.figure(figsize=(8, 5))
models = ['Naive Bayes', 'Logistic Regression']
accuracy = [accuracy_score(y_test, ŷ_test_nb), accuracy_score(y_test, ŷ_test_lr)]
sns.barplot(x=models, y=accuracy, palette='Set1')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()
