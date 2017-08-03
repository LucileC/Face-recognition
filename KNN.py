# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:58:22 2017

@author: luluc
"""

from sklearn.neighbors import KNeighborsClassifier
from os import walk
import cv2
from sklearn.metrics import accuracy_score, make_scorer
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import shuffle

males = ['Cary Elwes', 'Chris Klein', 'Chris Noth', 'Gary Dourdan', 'Jason Statham', 'Keer Smith', 'Leonardo DiCaprio', 'Matt Long', 'Paul Walker', 'Richard Madden', 'Gerard Butler', 'Michael Vartan',  'Daniel Radcliffe']
females =  ['Carmen Electra', 'Dana Delany', 'Didi Conn', 'Dina Meyer', 'Holly Marie Combs', 'Kim Cattrall', 'Laura Innes', 'Lindsay Hartley', 'Loni Anderson', 'Teri Hatcher', 'Vanessa Marcil', 'Angie Harmon', 'Lorraine Bracco', 'Peri Gilpin']
 
def getNames(filename):
    l = [x for x in map(str.strip, filename.split('.')) if x]
    file_name_wout_ext = l[0]
    l = [x for x in map(str.strip, file_name_wout_ext.split('_')) if x]
    actor_name = l[0]
    return file_name_wout_ext, actor_name

def getGender(actorname):
    return "male" if (actorname in males) else "female"

def getSet(folder=None, ytype = "Name"):
#    X=np.matrix((np.ones(256))).transpose()
    X = list()
    y = list()
    f = []
    path = "resized_faces"+ ( "/"+folder if (folder) else "" ) 
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
#        break
    shuffle(f)
    for file in f:
        #get file_name without jpg extension
        file_name, actor_name = getNames(file)
        imgname = path+"/"+file
        img = cv2.imread(imgname,0)
        hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
        X.append(hist)
        if ytype == "Name":
            y.append(actor_name)
        elif ytype == "Gender":
            gender = getGender(actor_name)
            y.append(gender)
    X = np.matrix(X)
    n,m = X.shape
    X = np.column_stack((np.ones(n),X))
    return X,y


def getAccuracyScores(clf, X_train ,y_train, X_cv, y_cv, X_test, y_test, min_training_ex = 25, max_training_ex = 600, step = 25):
    accuracy_scores_train = list()
    accuracy_scores_cv = list()
    accuracy_scores_test = list()
    for i in range(min_training_ex,max_training_ex,step):
        clf.fit(X_train[:i,:],y_train[:i])
        y_pred_train = clf.predict(X_train[:i,:])
        y_pred_cv = clf.predict(X_cv)
        y_pred_test = clf.predict(X_test)
        accuracy_scores_train.append(accuracy_score(y_train[:i],y_pred_train))
        accuracy_scores_cv.append(accuracy_score(y_cv,y_pred_cv))
        accuracy_scores_test.append(accuracy_score(y_test,y_pred_test))
    return accuracy_scores_train, accuracy_scores_cv, accuracy_scores_test

############################################################################
##   Face recognition / gender recognition: READING DATA / BUILDING SETS  ##
############################################################################
    
#building the sets
X_train, y_train = getSet(folder="training",ytype="Gender")
X_cv, y_cv = getSet(folder="cv",ytype="Gender")
X_test, y_test = getSet(folder="test",ytype="Gender")
X = np.concatenate((np.concatenate((X_train, X_cv), axis=0),X_test), axis=0)
y = list(y_train)
y.extend(y_cv)
y.extend(y_test)


############################################################################
##       Face recognition / gender recognition: TRAiNING CLASSIFIER       ##
############################################################################

#using cross-validation set to compute the best K
best_classifier = None
best_accuracy = 0
for i in range(3,20):
    classifier = KNeighborsClassifier(n_neighbors = i, metric = "minkowski", p = 2)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_cv)
    accuracy = accuracy_score(y_cv,y_pred)
    print (i.__str__()+": "+accuracy.__str__())
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = classifier

#getting accuracy on test set
y_pred = best_classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


############################################################################
##       Face recognition / gender recognition: LEARNING CURVES           ##
############################################################################

#ploting learning curves myself of the predefined traning, cv and test sets
acc_scores_train, acc_scores_cv, acc_scores_test = getAccuracyScores(best_classifier, X_train, y_train, X_cv, y_cv, X_test, y_test)
plt.grid()
plt.title("Leraning curve of KNN algorithm for image recognition")
plt.xlabel("Number of training examples")
plt.ylabel("Accuracy core")
plt.plot(range(25,600,25), acc_scores_train, 'o-', color="r", label="Training score")
plt.plot(range(25,600,25), acc_scores_cv, 'o-', color="b", label="CV score")
plt.plot(range(25,600,25), acc_scores_test, 'o-', color="g", label="Test score")
plt.legend(loc="best")
plt.show()

#plotting learning curves with the learning_curves fct from sklearn.model_selection 
train_sizes, train_scores, test_scores = learning_curve(best_classifier, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=None, scoring=make_scorer(accuracy_score), exploit_incremental_learning=False, n_jobs=1, pre_dispatch="all", verbose=0)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.title("Leraning curve of KNN algorithm for image recognition")
plt.xlabel("Number of training examples")
plt.ylabel("Accuracy core")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Test score")
plt.legend(loc="best")
plt.show()


############################################################################
##              Face recognition: FAILURE CASES STUDY                     ##
############################################################################

f =[]
for (dirpath, dirnames, filenames) in walk("resized_faces/training"):
    f.extend(filenames)
    
def getImage(index):
    imgname = f[index] #if len(f) > index else None
    return imgname,'resized_faces/training/'+imgname

failure_cases = ["Angie Harmon_166","Angie Harmon_168","Daniel Radcliffe_144","Daniel Radcliffe_145","Daniel Radcliffe_146"]
for failure_case in failure_cases:
    path = "resized_faces/test/"+failure_case+".png"
    img = cv2.imread(path,0)
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
    X = np.matrix([hist])
    X = np.column_stack((np.ones(1),X))
    y_pred2 = best_classifier.predict(X)
    print(failure_case+", is mistaken with "+y_pred2[0]+", with nearest neighbors: ")
    neighbors = best_classifier.kneighbors(X, 7, return_distance=False)
    
    images = [mpimg.imread(path)]
    for idx in neighbors[0]:
        filename, path = getImage(idx)
        _, actorname = getNames(filename)
        print(filename)
        images.append(mpimg.imread(path))
    plt.figure(figsize=(6,6))
    columns = 8
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)
        plt.axis('off')


############################################################################
##                Gender recognition: TEST WITH NEW ACTORS                ##
############################################################################
X_test, y_test = getSet(folder="genderTest",ytype="Gender")
y_pred = best_classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)