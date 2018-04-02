# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:52:38 2017

@author: admin
"""

import csv
import numpy as np
#import matplotlib.pyplot as plt
def inputfeat(filename):

    labels=[]
    list1=[]
    A=0
    A1=0
    ind=[]
    V=[]
    with open(filename, 'r',encoding="utf8") as csv1:
  
         reader=csv.reader(csv1)
         for row in reader:
             list1.append(row)
             for i in row:
                 A=A+1
                 O=i.split(' ')
                 Y=len(O);
                 if O[0] == '-1':
                    labels.append(-1)
                 else:
                    labels.append(1)
                 for j in range(1,len(O)):
                     A1=O[j]
                     Dat=A1.split(':')
                 
                     ind.append(Dat[0])
                     V.append(Dat[1])
                 
    Val=np.array(list(V),dtype=float)
    ind1=np.array(list(ind),dtype=int)
    labels1=np.array(list(labels),dtype=int)
    col=len(ind1)//len(list1)
    Val1=np.reshape(Val,(len(list1),col))
    ind2=np.reshape(ind1,(len(list1),col))
    length=np.max(ind2[:,-1])+1
    if length<69:
       length=length+1
    

    X=np.zeros((len(list1),length))
    newX=np.zeros((len(list1),length+1))
    for i in range(0,ind2.shape[0]):
        for j in range(0,ind2.shape[1]):
            f=ind2[i][j]
            f1=f-1
            X[i][f1]=Val1[i][0]
        X[i][-1]=1
    
    
    for i in range(0,ind2.shape[0]):
        for j in range(0,ind2.shape[1]):
            f=ind2[i][j]
            f1=f-1
            newX[i][f1]=Val1[i][0]
        newX[i][-2]=1
    newX[:,-1]=labels1
    
    return(newX,X,labels1,length)
    
    
def train(epochs,r,a,newX,ct):

    

    for j in range(epochs):
        w=a
        for i in range(0,len(newX)):
#            a=a+w
            if (newX[i][-1]*(np.dot(w,newX[i][0:-1])))<=0:
               w[0:-1]=w[0:-1]+r*newX[i][-1]*newX[i][0:-2]
               w[-1]=w[-1]+r*newX[i][-1]
            
               ct=ct+1
        a=a+w
        np.random.shuffle(newX)
    return(a,ct)
    
def predict(w,newX):
    predicted_output=np.zeros(len(newX))
    for i in range(0,len(newX)) :        
        if (np.dot(w,newX[i][0:-1]))>=0:
            predicted_output[i]=1
        else:
            predicted_output[i]=-1
    return(predicted_output)
    
newX,X,labels,length=inputfeat('phishing.train')
epochs=10
r=0.1
length=X.shape[1] 

      
np.random.seed(2)          
w=np.random.uniform(-0.01,0.01,[length])    
np.random.seed(2)   
v=np.random.uniform(-0.01,0.01,[length])
ct=0
w1,ct=train(epochs,r,w,newX,ct)
newX1,X1,labels1,length1=inputfeat('phishing.test')
newX2,X2,labels2,length2=inputfeat('phishing.dev')

predicted_output_dev=np.zeros(len(newX2))
predicted_output_dev=predict(w1,newX2)
A2=0
for i in range(0,len(predicted_output_dev)):
    if predicted_output_dev[i]==labels2[i]:
        A2=A2+1
acc_dev=A2/len(predicted_output_dev)*100
print('Accuracy for testing on development set\t:',acc_dev)
print('Number of updates:',ct)
'''
Plotting learning curve for development set
'''
def traindev(epochs,r,a,newX,ct,newX2,labels2):
    w_1=np.zeros((epochs,length2))
    
    
    for j in range(epochs):
        cnt=0
        w=a
        for i in range(0,len(newX)):
#            a=a+w
            if (newX[i][-1]*(np.dot(w,newX[i][0:-1])))<=0:
               w[0:-1]=w[0:-1]+r*newX[i][-1]*newX[i][0:-2]
               w[-1]=w[-1]+r*newX[i][-1]
               ct=ct+1
        a=a+w   
        w_1[j,:]=a   
        dev=predict(a,newX2)
        for k in range(0,len(dev)):
            
            if dev[k]==labels2[k]:
               cnt=cnt+1
            acc_de=cnt/len(dev)*100
            
        acc_deV[j]=acc_de
        np.random.shuffle(newX)
    return(w_1,acc_deV)
np.random.seed(2)          
w=np.random.uniform(-0.01,0.01,[length])   
ct=0    
acc_deV=np.zeros(epochs)
w_1,acc_deV=traindev(epochs,r,w,newX,ct,newX2,labels2)    
#plt.plot(range(epochs),acc_deV)
#plt.xlabel('epochs')
#plt.ylabel('Accuracy')
#plt.show()      


for i in range(0,len(acc_deV)):
    val=max(acc_deV)
    if acc_deV[i]==val:
       ind=i
       break
output1=predict(w_1[ind],newX1)
cnt1=0
for k in range(0,len(output1)):
            
            if output1[k]==labels1[k]:
               cnt1=cnt1+1
            acc_test=cnt1/len(output1)*100        
        
print('Accuracy of testing data:',acc_test)      


