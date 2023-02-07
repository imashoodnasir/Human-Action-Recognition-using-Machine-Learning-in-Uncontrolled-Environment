import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

from sklearn.metrics import accuracy_score , confusion_matrix , precision_score
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

d = lambda x1,y1,z1,x2,y2,z2 : ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5  # lambda function to find euclidean distance

# This function is to trim the total no of frames to 40 frames , which are selected equi-distance from the total frames
def trim(x,minvalue):
    r = (x.shape[2]/(minvalue) )
    array = []
    t= 0 + r
    i=int(t)
    while i+1 < x.shape[2]:
        array.append( x[:,:,i] + ( (x[:,:,i] - x[:,:,i+1])*(t-i) )  )
        t+=r
        i=int(t)-1
    return(np.array(array)[:40])  #performed slicing operation to restrict size . Since few objects has output length to be 41

file_names =os.listdir('../input/skeleton')  # get all the file names in the Dataset Directory.
file_names[:10]

# we are importing the dataset from the matlab format to numpy arrays that is recoganised by python.
no_of_frames =  []
train        =  []
train_labels =  []
test         =  []
test_labels  =  []
for i in file_names:
    if ".txt" not in i : 
        if "s2" in i or "s4" in i or "s6" in i or "s8" in i:
            train.append(loadmat(os.path.join("../input/skeleton/", i))['d_skel'])
            train_labels.append(int(i.split('_')[0].replace('a','')))
        else:
            test.append(loadmat(os.path.join("../input/skeleton/", i))['d_skel'])
            test_labels.append(int(i.split('_')[0].replace('a','')))    
        no_of_frames.append( loadmat(os.path.join("../input/skeleton/", i))['d_skel'].shape[2] )
        


minvalue = min(no_of_frames)   # assigning the min value as a global variable

print("min :", min(no_of_frames) ,"\t" , file_names[ no_of_frames.index( min(no_of_frames)) ] ,"\t", no_of_frames.index(min(no_of_frames)) ) 
print("max :", max(no_of_frames) ,"\t" , file_names[ no_of_frames.index( max(no_of_frames)) ] ,"\t", no_of_frames.index(max(no_of_frames)) )
print("length Of Dataset (Train and Test) : " , len(train) ,"  ", len(test))
print("Dimension of a single dataset : ", train[0].shape )


# Trimming the frames in each video to be 40 frames with intropolation therefore we can reduce the data loss.
temp=[]
for i in train:
    temp.append(trim(i,minvalue))
#     c.append (trim(i,minvalue).shape[0] )
train = np.array(temp)
print(train.shape)

temp=[]
for i in test:
    temp.append(trim(i,minvalue))
#     c.append (trim(i,minvalue).shape[0] )
test = np.array(temp)
print(test.shape)


train_dist_matrix=[] 
for i in train:
    train_dist_matrix.append( d(i[:,:,0].transpose(),i[:,:,1].transpose(),i[:,:,2].transpose(),i[:,1,0],i[:,1,1],i[:,1,2])  )
train_dist_matrix=np.array(train_dist_matrix)
print (train_dist_matrix.shape )

test_dist_matrix=[] 
for i in test:
    test_dist_matrix.append( d(i[:,:,0].transpose(),i[:,:,1].transpose(),i[:,:,2].transpose(),i[:,1,0],i[:,1,1],i[:,1,2])  )
test_dist_matrix=np.array(test_dist_matrix)
print (test_dist_matrix.shape )


# Converting the data into 2 Dimension ,Since it is easier to train models with 2 Dimensions.
temp =[]
for i in train_dist_matrix:
    temp.append(i.transpose().flatten())
train_dist_matrix= np.array(temp)
print(train_dist_matrix.shape)
temp =[]
for i in test_dist_matrix:
    temp.append(i.transpose().flatten())
test_dist_matrix= np.array(temp)
print(test_dist_matrix.shape)


label_predicted=[]
model= knn(n_neighbors=len(set(train_labels)))

model.fit(train_dist_matrix , train_labels)
label_predicted = model.predict(test_dist_matrix)




print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))




label_predicted=[]
model= MultinomialNB()

model.fit(train_dist_matrix , train_labels)
label_predicted = model.predict(test_dist_matrix)

print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))



label_predicted=[]
regressor = LogisticRegression(solver="liblinear")

regressor.fit(train_dist_matrix,train_labels)
label_predicted = regressor.predict(test_dist_matrix)



print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))




label_predicted=[]
regressor = LogisticRegression(solver="newton-cg")

regressor.fit(train_dist_matrix,train_labels)
label_predicted = regressor.predict(test_dist_matrix)



print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))




label_predicted=[]
regressor = LogisticRegression(solver="sag")

regressor.fit(train_dist_matrix,train_labels)
label_predicted = regressor.predict(test_dist_matrix)



print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))



label_predicted=[]
regressor = LogisticRegression(solver="saga")

regressor.fit(train_dist_matrix,train_labels)
label_predicted = regressor.predict(test_dist_matrix)



print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))




label_predicted=[]
Regressor = LogisticRegression(solver="newton-cg")
Scaler =StandardScaler()

Regressor.fit( Scaler.fit_transform(train_dist_matrix)  ,train_labels)
label_predicted = regressor.predict( Scaler.transform(test_dist_matrix))



print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))



label_predicted=[]
Regressor = LogisticRegression(solver="saga")
Scaler =StandardScaler()

Regressor.fit( Scaler.fit_transform(train_dist_matrix)  ,train_labels)
label_predicted = regressor.predict( Scaler.transform(test_dist_matrix))



print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))



label_predicted=[]
Regressor = LogisticRegression(solver="sag")
Scaler =StandardScaler()

Regressor.fit( Scaler.fit_transform(train_dist_matrix)  ,train_labels)
label_predicted = regressor.predict( Scaler.transform(test_dist_matrix))



print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))



label_predicted=[]
Regressor = LogisticRegression(solver="liblinear")
Scaler =StandardScaler()

Regressor.fit( Scaler.fit_transform(train_dist_matrix)  ,train_labels)
label_predicted = regressor.predict( Scaler.transform(test_dist_matrix))




print("Accuracy : ", accuracy_score(label_predicted, test_labels))
print("Pression Score : \n", precision_score(test_labels,label_predicted,average=None))

print("Confusion Matrix")
sns.heatmap(confusion_matrix(label_predicted, test_labels))



