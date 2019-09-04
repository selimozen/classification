#Sınıflandırmada kullanılcak datalar...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Iris.csv')
data.drop('Id', axis = 1, inplace = True)

#sınıflandırılcak datanın kategorikleştirilmesi
species = data.iloc[:, 4:5]
data.drop('Species', axis = 1, inplace = True)
le = LabelEncoder()
species = le.fit_transform(species)

#na verileri düzelmte
data = data.fillna(data.mean())
species = pd.DataFrame(data = species, index = range(150), columns = ['species'])
species = species.fillna(species.mean())

#dataları bölme

x_train,x_test,y_train,y_test = train_test_split(data, species, test_size = 0.33, random_state = False)



from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.plot(y_test,y_pred, 'blue')

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')

svc.fit(x_train,y_train)
y_pred2 = svc.predict(x_test)

plt.subplot(2,1,1)
plt.plot(y_pred,y_test, 'blue')
plt.subplot(2,1,2)
plt.plot(y_pred2,y_test, 'purple')
