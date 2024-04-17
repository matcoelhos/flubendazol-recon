from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import time
import csv
import cv2
import pandas as pd

violations_list = ['cristalino','moinho','evaporacao']

test_dataset = pd.read_csv('test.csv')
x_test = test_dataset.iloc[:, :-1].values
y_test = test_dataset.iloc[:, -1].values

# Recreate the exact same model, including its weights and the optimizer
dnn_model = tf.keras.models.load_model('dnn_model.h5')
dnn_model.summary()

predictions = []
i = 0
length = len(y_test)
for i in range(length):
	x = x_test[i,:]
	x = np.reshape(x, (1,x.shape[0]))
	P = dnn_model(x)
	pred = np.argmax(P[0])
	predictions.append(pred)
	print("%d/%d"%(i,length),end='\r')
	#input()
predictions = np.array(predictions)
print()

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
i = 0
for violation in violations_list:
	print(i,violation)
	i+=1
print(classification_report(y_test, predictions, labels=range(len(violations_list)), target_names=violations_list))
print(confusion_matrix(y_test, predictions))
print('Acc: %.2f %c'%(100*accuracy_score(y_test, predictions),'%'))
print(predictions.shape,y_test.shape)