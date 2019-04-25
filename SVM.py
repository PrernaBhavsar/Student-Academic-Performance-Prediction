
# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('edu.csv')
X = dataset.iloc[:, 0:16 ].values
y = dataset.iloc[:, 16].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X[:,2]=labelencoder_X.fit_transform(X[:,2])
X[:,3]=labelencoder_X.fit_transform(X[:,3])
X[:,4]=labelencoder_X.fit_transform(X[:,4])
X[:,5]=labelencoder_X.fit_transform(X[:,5])
X[:,6]=labelencoder_X.fit_transform(X[:,6])
X[:,7]=labelencoder_X.fit_transform(X[:,7])
X[:,8]=labelencoder_X.fit_transform(X[:,8])
X[:,13]=labelencoder_X.fit_transform(X[:,13])
X[:,14]=labelencoder_X.fit_transform(X[:,14])
X[:,15]=labelencoder_X.fit_transform(X[:,15])

onehotencoder= OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,7,8,13,14,15])
X= onehotencoder.fit_transform(X).toarray()


labelencoder_y= LabelEncoder()
y= labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.2, random_state= 0) 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train) 
X_test=sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 
cm = confusion_matrix(y_test, y_pred)
Accuracy_score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", Accuracy_score*100)
print("Misclassification Score:", 100-(Accuracy_score*100))
print('Precision Score:',precision_score(y_test, y_pred,average='weighted'))
print('Recall Score:',recall_score(y_test, y_pred, average='weighted'))
print('F1 Score:',f1_score(y_test, y_pred, average='weighted'))

