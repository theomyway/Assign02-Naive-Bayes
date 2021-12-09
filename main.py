#Omar's Contribution

#Mounting Our Drive On Cloab
from google.colab import drive
drive.mount('/content/drive')

#importing different libraries to work with
import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#Loading Training Data
train=panda.read_csv('/content/drive/MyDrive/smtp/train.csv')

#Using variable for labels
y = train.Cover_Type

#For functions
X = train.drop('Cover_Type', axis=1)

#Splitting The Data y 20% and x 80%
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

#Hamza's Contribution
#Using No Smoothing:
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
NoMNB=clf.score(t_test,y_test)
print("Accuracy Score - NoMNB",NoMNB*100)


#Using Laplace Smoothing with Alpha = 1
clf = MultinomialNB(alpha=1)
clf.fit(abs(t_train),y_train)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LPMNB=clf.score(t_test,y_test)
print("Accuracy Score - LPMNB",LPMNB*100)

#Using Lidstone  Smoothing with Alpha = 0.5
clf = MultinomialNB(alpha=0.5)
clf.fit(abs(t_train),y_train)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LDMNB=clf.score(t_test,y_test)
print("Accuracy Score - LDMNB",LDMNB*100)


#Loading Training Data
test=panda.read_csv('/content/drive/MyDrive/smtp/test.csv')
test.head()


#Using No Smoothing:
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
Cover_type=clf.predict(test)
print("Predicted Values",Cover_type)

#Hamza's Part
#Exporting The Id And Cover_Type Columns Into Sample Csv
sample = test[['Id']].copy()
sample['Cover_Type'] = Cover_type
print(sample)


#Creating csv to submit on kaggle
sample.to_csv('sample.csv',index=False)
