import pandas as pd 
import os
import scipy.io
#import wfdb
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import signal
#create a list of every file in Data that ends in mat 
fileList =[]
headerList =[]
basePath="./Data"
for path in os.listdir(basePath):
    if ".mat" in path:
        fileList.append(path)
    if ".hea" in path:
        headerList.append(path)

#lets take a look at one piece of data in this set looks like 
for file in fileList:
    matFile = scipy.io.loadmat(f'{basePath}/{file}')
    break

diagnosisCount = {}
#i want to make a count of each Dx
for header in headerList:
    hea_file_path = f"{basePath}/{header}"
    with open(hea_file_path,"r") as file:
        for line in file:
            if line.startswith("#Dx"):
                diag = line.split(":")[1].strip()
                diagList = diag.split(",")
                for diagnosis in diagList:
                    if diagnosis not in diagnosisCount.keys():
                        diagnosisCount[diagnosis] =1
                    else:
                        diagnosisCount[diagnosis] +=1

relevant =[]
for key in diagnosisCount.keys():
    if diagnosisCount[key] >750:
        relevant.append(key)

diagnosticLookup = {
    '427084000': 'Chronic atrial fibrillation',
    '164934002': 'Hypertension',
    '426783006': 'Heart failure',
    '67741000119109': 'Chronic systolic heart failure',
    '428750005': 'Congestive heart failure',
    '111975006': 'Atrial fibrillation',
    '164930006': 'Chronic heart failure',
    '59931005': 'Myocardial infarction',
    '426177001': 'Coronary artery disease',
    '164873001': 'Cardiomyopathy',
    '425623009': 'Hypertensive heart disease',
    '284470004': 'Ischemic heart disease',
    '39732003': 'Hypertensive disease',
    '270492004': 'Hypertensive heart disease'
}

relevantDataFiles =[]
DataFileDiagnosis = []
for header in headerList:
    hea_file_path = f"{basePath}/{header}"
    with open(hea_file_path,"r") as file:
        Matfile = None
        for line in file:
            if line.startswith("E") and Matfile == None:
                Matfile = line.split(" ")[0]
            if line.startswith("#Dx"):
                diag = line.split(":")[1].strip()
                diagList = diag.split(",")
                for diagnosis in diagList:
                    if diagnosis in diagnosticLookup.keys():
                        relevantDataFiles.append(Matfile)
                        DataFileDiagnosis.append(diagnosticLookup[diagnosis])
                        break    
                    else:
                        diagnosisCount[diagnosis] +=1


#Now we have a list of files and there diagnosis

#DataFile Diagnosis is effectivley or Y 
#or should we go the route of putting this all in a DF and using Split functions from Sklear to match what is

X=[] 
#max_length =5000
for file in relevantDataFiles:
    matFile = scipy.io.loadmat(f'{basePath}/{file}')
    tempData=(matFile['val'][:,0])
    #dataPadded = np.pad(tempData, (0, max_length - len(tempData)), mode='constant', constant_values=0)
    dataDownSampled = signal.resample(tempData, 50)
    X.append(dataDownSampled)
#convert them into np.arrays
X = np.array(X)
y= np.array(DataFileDiagnosis)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("training classifer")
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



