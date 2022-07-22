import sklearn.metrics as metrics
import pandas as pd
import seaborn as sns
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
dataDir=["D:\\S8\\Project\\fiNAL\\dataset\\testDataset\\NORMAL","D:\\S8\\Project\\fiNAL\\dataset\\testDataset\\PNEUMONIA"]
fileCounter=0
trueCounter=0
FN=0
FP=0
TP=0
TN=0
yTest=[]
prediction=[]
classifier=load_model('./resources/model/dcdis.h5')
fileNameList=[]

truePositive=[]
falsePositive=[]
tpr1=[]
fpr1=[]
for data_dir in dataDir:
    for fileName in os.listdir(data_dir):
        print('Taking Images .......'+str(fileCounter))

        fileActual=os.path.join(data_dir,fileName)
       
        test_image = image.load_img(fileActual,target_size = (128, 128))
        fileCounter+=1
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        if data_dir.split('\\')[-1]=='NORMAL':
            yTest.append(0)
        else:
            yTest.append(1)
        if result[0][0] == 0:
            if data_dir.split('\\')[-1]=='NORMAL':
                TN+=1
            else:
                FN+=1
            prediction.append(0)
            trueCounter+=1
        else:
            if data_dir.split('\\')[-1]=='PNEUMONIA':
                TP+=1
            else:
                FP+=1
            prediction.append(1)
        truePositive.append(TP+FN)
        falsePositive.append(TN+FP)
        try:
            tpr1.append(TP/TP+FN)
            fpr1.append(TN/TN+FP)
        except ZeroDivisionError:
            pass
       
labels=['Normal','Pneumonia']
confusionMatrix=np.array([[TN,FP],[FN,TP]])
confusionMatrix = pd.DataFrame(confusionMatrix , index = ['0','1'] , columns = ['0','1'])
   
truePositiveRate = TP / (TP + FN)
sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
falsePositiveRate = 1 - Specificity
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Accuracy=(TN+TP)/(TP+TN+FP+FN)
f1Score=2*((Precision*Recall)/(Precision+Recall))
print("specificity :",Specificity)
print("Sensitivity :",sensitivity)
print("Recall      :",Recall)
print("Precision   :",Precision)
print("Accuracy    :",Accuracy)
print("F1-Score    :",f1Score)
fpr, tpr, threshold = metrics.roc_curve(yTest, prediction)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.figure(figsize = (10,10))
sns.heatmap(confusionMatrix,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
