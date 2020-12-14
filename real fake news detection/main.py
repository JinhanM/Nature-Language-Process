import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

real['Label'] = 0
fake['Label'] = 1

df = pd.concat([fake, real], axis=0)

import re


def clean(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    # New Feature: Clean multiple spaces
    text = re.sub(' +', ' ', text)

    return text


df['text'] = df['text'].apply(clean)


from sklearn.model_selection import KFold, cross_val_score, train_test_split


x=df['text']
y=df['Label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

model = QDA()
model.fit(xv_train.toarray(), y_train)
print(model.score(xv_test.toarray(), y_test))
