
#importing libraries panda and skikit 
import pandas as panda
from sklearn.model_selection import train_test_split

#Getting the whole data set by mentioning its path of drive 
train=panda.read_csv('/content/drive/MyDrive/smtp/train.csv')
train.head()


#varibale for labels
y = train.Cover_Type

#for functions
X = train.drop('Cover_Type', axis=1)

#splitting the data  in x and y 80% and 20%
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

#getting the splitting data
print("\nt_train:\n")
t_train.head()

print("\nt_test:\n")
t_test.head()
#for no smoothing using linear
from sklearn.linear_model import LinearRegression
mnb = LinearRegression()
mnb.fit(t_train,t_train)
mnb.predict(t_test)
#printing efficiency score
mnb.score(t_test,y_test)
