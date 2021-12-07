 # linear algebra
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pickle
import os

df = pd.read_csv('suspicious tweets.csv')

#top n (5 by default) rows of a data frame or series. 
df.head(10)

#Return a tuple representing the dimensionality of the DataFrame
df.shape

#Detect missing values.
df.isnull().sum()

# Generate descriptive statistics.
# Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values
df.describe()

# Print a concise summary of a DataFrame
df.info()

# Hash table-based unique. Uniques are returned in order of appearance. 
df['label'].unique()

# Return a Series containing counts of unique rows in the DataFrame.
df['label'].value_counts()

df.groupby('label').describe()

# genertaing length of message column using len funtion
df['length'] = df['message'].apply(len)
df.head()

# Plot pairwise relationships in a dataset.
# Create the default pairplot
sns.pairplot(df)

# A correlation matrix is a matrix that shows the correlation values of the variables in the dataset.
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):
    df.dataframeName = 'suspicious tweets.csv'
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
# Correlation matrix:
plotCorrelationMatrix(df, 8)

# counts of observations in each categorical bin using bars
# ax = sns.countplot(x="class", data=suspicious tweets)
sns.countplot(df['label'])

count_Class = pd.value_counts(df.label, sort = True)

#Data to PLot
labels = '0','1'
sizes = [count_Class[0], count_Class[1]]
colors = ['red','blue']
explode = (0.1, 0.1)

#Plot
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%', shadow = True, startangle = 90)
plt.title('Percentage of 0s and 1s in column label')
plt.axis('equal')
plt.show()

# This transformer should be used to encode target values, i.e. y, and not the input X.
# Transform labels to normalized encoding.
label = LabelEncoder()
df['label'] = label.fit_transform(df['label'])

X = df['message']
y = df['label']

ps = PorterStemmer()
corpus = []

for i in range(len(X)):
    print(i)
    review = re.sub("[^a-zA-Z]"," ", X[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
X.shape

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=101)

X_train.shape , X_test.shape , y_train.shape , y_test.shape

mnb = MultinomialNB()
mnb.fit(X_train , y_train)

pred = mnb.predict(X_test)
print(np.mean(pred == y_test))

print(accuracy_score(y_test , pred))
print(confusion_matrix(y_test , pred))
print(classification_report(y_test , pred))

pd.DataFrame(np.c_[y_test , pred] , columns=["Actual" , "Predicted"])

pickle.dump(cv , open("count-Vectorizer.pkl" , "wb"))
pickle.dump(mnb , open("Cyberbullying_Detection_One.pkl" , "wb"))  # 1: pos , 0:Neg

save_cv = pickle.load(open('count-Vectorizer.pkl','rb'))
model = pickle.load(open('Cyberbullying_Detection_One.pkl','rb'))