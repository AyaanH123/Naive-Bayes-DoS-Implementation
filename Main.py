import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from ctypes import *

df = pd.read_csv("DoS_dataset.csv")
    
training_length = len(df)*0.67
training_data = df.iloc[:int(training_length),:]
test_data = df.loc[int(training_length)+1:,:]

classifier = GaussianNB()
classifier.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])

predicted_labels = classifier.predict(test_data.iloc[:,:-2])

expected_labels = test_data.iloc[:,-1]

print(expected_labels," ", predicted_labels)

accuracy = classifier.score(test_data.iloc[:,:-2], expected_labels)

print(accuracy)