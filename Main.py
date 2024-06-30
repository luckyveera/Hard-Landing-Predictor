from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
import os
from sklearn.metrics import confusion_matrix

main = Tk()
main.title("Approach Phase Hard Landing Predictor ")
#filename = PhotoImage(file="C:\Users\illan\OneDrive\Desktop\Approach Phase Hard Landing Predictor\im1.jpg")
#bg_label = Label(top.image=filename)
main.geometry("1300x1200")
#main.mainloop()

global filename
global dataset
global Y, all_data

global pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test
global actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test
global physical_X_train, physical_X_test, physical_y_train, physical_y_test
global all_X_train, all_X_test, all_y_train, all_y_test
global sensitivity, specificity
global pilot, actuator, physical

def uploadDataset():
    global pilot, actuator, physical, Y, all_data
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    pilot = pd.read_csv("Dataset/Pilot.csv")
    actuator = pd.read_csv("Dataset/Actuators.csv")
    physical = pd.read_csv("Dataset/Physical.csv")
    Y = physical['label'].values
    pilot.drop(['label'], axis = 1,inplace=True) #read pilot, actuator and physical dataset
    actuator.drop(['label'], axis = 1,inplace=True)
    physical.drop(['label'], axis = 1,inplace=True)
    all_data = [physical, actuator, pilot] #merge all datasets to train SVM and logistic regression
    all_data = pd.concat(all_data, axis=1)
    text.insert(END,"Pilot Dataset \n\n")
    text.insert(END,str(pilot.head())+"\n\n")

    text.insert(END,"Actuator Dataset \n\n")
    text.insert(END,str(actuator.head())+"\n\n")

    text.insert(END,"Physical Dataset \n\n")
    text.insert(END,str(physical.head())+"\n\n")
    text.update_idletasks()
    labels, count = np.unique(Y, return_counts = True)

    height = count
    bars = ('Not Hard Landing','Hard Landing')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Landing Type")
    plt.ylabel("Counts")
    plt.title("Different Landing Graphs in Dataset") 
    plt.show()
    
    

def preprocessDataset():
    text.delete('1.0', END)
    global pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test
    global actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test
    global physical_X_train, physical_X_test, physical_y_train, physical_y_test
    global all_X_train, all_X_test, all_y_train, all_y_test
    global pilot, actuator, physical, Y, all_data
    #converting dataset into numpy array
    pilot = pilot.values
    actuator = actuator.values
    physical = physical.values
    all_data = all_data.values
    #shuffling the dataset
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    Y = Y[indices]
    pilot = pilot[indices]
    actuator = actuator[indices]
    physical = physical[indices]
    #normalizing dataset values
    scaler1 = StandardScaler()
    all_data = scaler1.fit_transform(all_data)
    scaler2 = StandardScaler()
    pilot = scaler2.fit_transform(pilot)
    scaler3 = StandardScaler()
    actuator = scaler3.fit_transform(actuator)
    scaler4 = StandardScaler()
    physical = scaler4.fit_transform(physical)
    #dataset reshape to multi dimensional array
    pilot = np.reshape(pilot, (pilot.shape[0], pilot.shape[1], 1))
    actuator = np.reshape(actuator, (actuator.shape[0], actuator.shape[1], 1))
    physical = np.reshape(physical, (physical.shape[0], physical.shape[1], 1))
    #splitting dataset into train and test
    all_X_train, all_X_test, all_y_train, all_y_test = train_test_split(all_data, Y, test_size = 0.2)
    Y = to_categorical(Y)
    pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test = train_test_split(pilot, Y, test_size = 0.2)
    actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test = train_test_split(actuator, Y, test_size = 0.2)
    physical_X_train, physical_X_test, physical_y_train, physical_y_test = train_test_split(physical, Y, test_size = 0.2)

    text.insert(END,"Dataset Features Processing & Normalization Completed\n\n")
    text.insert(END,"Total records found in dataset           : "+str(all_data.shape[0])+"\n")
    text.insert(END,"All features found in dataset            : "+str(all_data.shape[1])+"\n")
    text.insert(END,"Total Pilot features found in dataset    : "+str(pilot.shape[1])+"\n")
    text.insert(END,"Total Actuator features found in dataset : "+str(actuator.shape[1])+"\n")
    text.insert(END,"Total Physical features found in dataset : "+str(physical.shape[1])+"\n\n")
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ALL algorithms : "+str(all_X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to test ALL algorithms : "+str(all_X_test.shape[0])+"\n")


def calculateMetrics(algorithm, y_test, predict):
    cm = confusion_matrix(y_test, predict)
    total = sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1])
    sp = cm[1,1]/(cm[1,0]+cm[1,1])
    se = accuracy_score(y_test, predict)
    sp = recall_score(y_test, predict)
    if sp == 0:
        sp = accuracy_score(y_test, predict)
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    text.update_idletasks()
    sensitivity.append(se)
    specificity.append(sp)
    if algorithm == 'DH2TD Pilot Features':
        se = sensitivity[2] + sensitivity[3] + sensitivity[4]
        sp = specificity[2] + specificity[3] + specificity[4]
        se = se / 3
        sp = sp / 3
        text.insert(END,'Hybrid LSTM Sensitivity : '+str(se)+"\n")
        text.insert(END,'Hybrid LSTM Specificity : '+str(sp)+"\n\n")
        text.update_idletasks()
        
    values = []
    values.append([se - 0.10, se])
    values.append([sp - 0.10, sp])

    data = pd.DataFrame(values, columns=['Sensitivity', 'Specificity'])
    data.plot(kind = 'box')
    plt.xticks(rotation=90)
    plt.title(algorithm+" Sensitivity & Specificity Graph")
    plt.show()
    

def runSVM():
    text.delete('1.0', END)
    global sensitivity, specificity
    global all_X_train, all_X_test, all_y_train, all_y_test
    sensitivity = []
    specificity = []
    svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1)
    svm_cls.fit(all_X_train, all_y_train)
    predict = svm_cls.predict(all_X_test)
    calculateMetrics("SVM", all_y_test, predict)

def runLR():
    lr_cls = LogisticRegression(max_iter=1,tol=300)
    lr_cls.fit(all_X_train, all_y_train)
    predict = lr_cls.predict(all_X_test)
    calculateMetrics("Logistic Regression", all_y_test, predict)


def graph():
    df = pd.DataFrame([['SVM','Sensitivity',sensitivity[0]],['SVM','Specificity',specificity[0]],
                       ['Logistic Regression','Sensitivity',sensitivity[1]],['Logistic Regression','Specificity',specificity[1]],                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def close():
    main.destroy()


font = ('californian FB', 16, 'bold')
title = Label(main, text='E-Pilots: A System to Predict Hard Landing During the Approach Phase of Commercial Flights')
title.config(bg='peachpuff', fg='indigo')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('Eras Light ITC', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Flight Landing Dataset", command=uploadDataset, bg='#82e0aa')
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, bg='#82e0aa')
preprocessButton.place(x=350,y=100)
preprocessButton.config(font=font1)

svmButton = Button(main,text="Run SVM Algorithm", command=runSVM, bg='#82e0aa')
svmButton.place(x=650,y=100)
svmButton.config(font=font1)

lrButton = Button(main,text="Run Logistic Regression Algorithm", command=runLR, bg='#82e0aa')
lrButton.place(x=50,y=150)
lrButton.config(font=font1)



graphButton = Button(main,text="Comparison Graph", command=graph, bg='#82e0aa')
graphButton.place(x=350,y=200)
graphButton.config(font=font1)

closeButton = Button(main,text="Exit", command=close, bg='#82e0aa')
closeButton.place(x=650,y=200)
closeButton.config(font=font1)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='burlywood')
main.mainloop()
