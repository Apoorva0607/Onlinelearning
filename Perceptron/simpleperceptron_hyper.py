# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:53:16 2017

@author: admin
"""
import csv
import numpy as np

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



 



'''
SImple perceptron
'''

def train(epochs,r,w,newX,ct):

    d=w.shape[0]

    for j in range(epochs):
        for i in range(0,len(newX)):
            if (newX[i][-1]*(np.dot(w,newX[i][0:-1])))<=0:
               w[0:-1]=w[0:-1]+r*newX[i][-1]*newX[i][0:-2]
               w[-1]=w[-1]+r*newX[i][-1]
            
               ct=ct+1
   
        np.random.shuffle(newX)
    return(w,ct)

            
def predict(w,newX):
    predicted_output=np.zeros(len(newX))
    for i in range(0,len(newX)) :        
        if (np.dot(w,newX[i][0:-1]))>=0:
            predicted_output[i]=1
        else:
            predicted_output[i]=-1
    return(predicted_output)


file0='training00.data'
file1='training01.data'
file2='training02.data'
file3='training03.data'
file4='training04.data'



def hyperp(file0,file1,file2,file3,file4,r,epochs):
    newX0,X0,labels0,axi0=inputfeat(file0)
    newX1,X1,labels1,axi1=inputfeat(file1)
    newX2,X2,labels2,axi2=inputfeat(file2)
    newX3,X3,labels3,axi3=inputfeat(file3)
    newX4,X4,labels4,axi4=inputfeat(file4)
    length=X0.shape[1] 
    predicted_output=np.zeros(len(newX4))        
    np.random.seed(2)          
    w=np.random.uniform(-0.01,0.01,[length])    
    np.random.seed(2)   
    v=np.random.uniform(-0.01,0.01,[length])
    ct=0
    w0,ct0=train(epochs,r,w,newX0,ct)
    w1,ct1=train(epochs,r,w0,newX1,ct)
    w2,ct2=train(epochs,r,w1,newX2,ct)
    w3,ct3=train(epochs,r,w2,newX3,ct)
    predicted_output=predict(w3,newX4)
    A1=0
    for i in range(0,len(predicted_output)):
        if predicted_output[i]==labels4[i]:
           A1=A1+1
    acc=A1/len(predicted_output)*100
    
    return(acc)


r_1=[1,0.1,0.01]
def disphyper(file0,file1,file2,file3,file4):
    r_1=[1,0.1,0.01]
#    I=input('no. of epochs') ### try for epoch 20 , gives a better
#    I1=np.array(list(I),dtype=int)
    Acc_r_1=np.zeros(3,dtype=float)
    Acc_R_1=np.zeros(3,dtype=float)
    
    for i in range(0,len(r_1)):
        acc4=hyperp(file0,file1,file2,file3,file4,r_1[i],10)       
        acc0=hyperp(file1,file2,file3,file4,file0,r_1[i],10)
        acc1=hyperp(file0,file2,file3,file4,file1,r_1[i],10)
        acc2=hyperp(file0,file1,file3,file4,file2,r_1[i],10)
        acc3=hyperp(file0,file1,file2,file4,file3,r_1[i],10)
        Acc_r_1[i]=(acc0+acc1+acc2+acc3+acc4)/5
       
   
    for i in range(0,len(r_1)):
        acc4=hyperp(file0,file1,file2,file3,file4,r_1[i],20)       
        acc0=hyperp(file1,file2,file3,file4,file0,r_1[i],20)
        acc1=hyperp(file0,file2,file3,file4,file1,r_1[i],20)
        acc2=hyperp(file0,file1,file3,file4,file2,r_1[i],20)
        acc3=hyperp(file0,file1,file2,file4,file3,r_1[i],20)
        Acc_R_1[i]=(acc0+acc1+acc2+acc3+acc4)/5
      
        
    return(Acc_r_1,Acc_R_1)

Acc_r_1,Acc_R_1=disphyper(file0,file1,file2,file3,file4)  
R1=max(Acc_r_1)
R2=max(Acc_R_1) 
print('r=1 \t Mean_Accuracy for 10 epochs\t',Acc_r_1[0])
print('r=0.1\t Mean_Accuracy for 10 epochs\t',Acc_r_1[1])
print('r=0.01\t Mean_Accuracy for 10 epochs \t',Acc_r_1[2])
for i in range(0,len(r_1)):
    if Acc_r_1[i]==R1:
       print('Best learning rate r:',r_1[i])
       print('max accuracy for best parameters for 10 epochs\t:',Acc_r_1[i])
       
print('r=1 \t Mean_Accuracy for 20 epochs:',Acc_R_1[0])
print('r=0.1\t Mean_Accuracy for 20 epochs:',Acc_R_1[1])
print('r=0.01\t Mean_Accuracy for 20 epochs:',Acc_R_1[2])
for i in range(0,len(r_1)):
    if Acc_R_1[i]==R2:
       print('Best learning rate r:',r_1[i])
       print('max accuracy for best parameters for 20 epochs\t:',Acc_R_1[i])