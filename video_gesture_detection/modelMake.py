import numpy as np

import pickle

from sklearn.metrics import accuracy_score                  #import acuuracy score to find accuracy of the model
from sklearn.ensemble import RandomForestClassifier         #import randomforest
from sklearn.model_selection import train_test_split        #here we will import the test_test_split  which slipt the the data to testing and training


data_dictory = pickle.load(open('./data.pickle', 'rb'))     #opening the file contains  data dimension and labels for specidicaton of directory 

data_pf = np.asarray(data_dictory['data'])
labels_pf = np.asarray(data_dictory['labels'])

x_train, x_test, y_train, y_test = train_test_split(data_pf, labels_pf, test_size=0.2, shuffle=True, stratify=labels_pf)

model = RandomForestClassifier()

model.fit(x_train, y_train)                 #train the model
y_predict = model.predict(x_test)           #no. of pridiction 

score = accuracy_score(y_predict, y_test)           #find the accuracy of the model
print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()