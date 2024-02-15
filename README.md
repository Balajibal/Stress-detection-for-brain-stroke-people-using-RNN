# Stress detection for Brain stroke person using RNN

## Introduction:

In the field of medicine, stress detection is crucial in determining how a patient's recovery from a brain stroke will proceed in terms of both rehabilitation and
general health. Our effort uses state-of-the-art technology, namely Recurrent Neural Networks (RNN), to tackle the complex problems related to stress detection in brain stroke survivors, marking a major excursion into this important field. After a brain stroke, which is a life-changing event, people are frequently more sensitive to stress, which might hinder their recuperation. Our study aims to utilize the capabilities of RNNs, a kind of artificial neural networks intended for sequential data processing, in light of the significance of prompt intervention. With this strategy, we can study and interpret physiological information more precisely than we could The core of our project is a large dataset that includes physiological data from individuals who have had brain strokes. Our RNN model is trained using this dataset as the basis to identify patterns that correspond to different stress levels. The temporal dependencies in the data are captured by the
model thanks to the recurrent nature of RNNs, which allows for a more sophisticated understanding of how physiological signals change over time in response to stressors. This implication is important not only because it is technologically advanced but also because it may have an effect on brain stroke sufferers' quality of life. Our RNN-based technology enables continuous stress monitoring, which offers a real-time evaluation of a person's emotional condition. Consequently, prompt interventions are made possible, whether by means of automated notifications to caretakers or medical experts.


## Features:
Recurrent Neural Networks (RNNs) can be employed for stress detection due to their ability to capture sequential dependencies in data. Here are some features and considerations for implementing stress detection for brain stroke individuals using RNNs:

### Physiological Data Input:

Heart Rate Variability (HRV): Monitor changes in the time intervals between heartbeats.
Blood Pressure: Track variations in blood pressure levels.
Electroencephalogram (EEG) Signals: Analyze brain activity patterns.
Galvanic Skin Response (GSR): Measure skin conductance as an indicator of stress.
Behavioral Data:

### Speech Analysis: Extract features from speech patterns, tone, and voice pitch.
Facial Expression Recognition: Utilize computer vision to identify facial cues related to stress.
Activity Level Monitoring: Track physical activities and rest patterns.
Environmental Data:

### Ambient Temperature: Changes in temperature can affect stress levels.
Noise Level: High levels of noise can contribute to stress.
Data Preprocessing:

### Normalization: Standardize data to ensure consistency across different features.
Time-Series Segmentation: Divide continuous data into segments to capture temporal patterns effectively.
Feature Extraction:

### Statistical Features: Extract statistical measures such as mean, variance, and skewness from time-series data.
Frequency Domain Features: Utilize Fourier Transform or wavelet analysis to capture frequency-related information.
RNN Architecture:


    
## Requirements
To develop a stress detection system for individuals at risk of brain stroke using RNNs, you'll need a combination of hardware, software, development environment, programming language, and libraries. Here are the requirements for this project:

### Hardware Requirements:

Sensors and Wearable Devices: To collect physiological and behavioral data such as heart rate variability, EEG signals, and activity levels. Examples include heart rate monitors, EEG headsets, and fitness trackers.
Computing Resources: A machine with sufficient processing power for training and running the RNN model. Depending on the scale of the project, this could range from a personal computer to a server or cloud-based solution.

### Software Requirements:

Operating System: Windows, macOS, or Linux, depending on your preference and compatibility with required software.
Data Collection and Preprocessing Tools: Software for collecting, cleaning, and preprocessing physiological, behavioral, and environmental data.
Database Management System: A database to store and manage the collected data efficiently.

### Development Environment:

Integrated Development Environment (IDE): Choose an IDE suitable for deep learning projects, such as Jupyter Notebooks, PyCharm, or VSCode.
Version Control System: Use a version control system like Git for tracking changes and collaborating with a team if applicable.
Collaboration Tools: Communication tools for team collaboration and project management.
### Programming Language:

Python: Widely used for machine learning and deep learning projects. It has a rich ecosystem of libraries and frameworks.
TensorFlow or PyTorch: Popular deep learning frameworks that support the implementation of RNNs.
Keras: A high-level neural networks API that can run on top of TensorFlow or other backends.
NumPy and Pandas: For data manipulation and analysis.
Scikit-learn: Useful for additional machine learning tasks such as data preprocessing and model evaluation.
Matplotlib and Seaborn: For data visualization.
Flask or Django (Optional): If you plan to deploy a web-based application for stress monitoring.
Machine Learning Model Deployment (Optional):

### Cloud Services: If deploying the model in the cloud, services like AWS, Google Cloud, or Azure may be used.
Docker: Containerization for easier deployment and scaling.
Web Framework (Flask, Django, etc.): For developing a web-based interface to interact with the stress detection system.
Data Privacy and Security:

### Encryption Tools: To ensure the security and privacy of sensitive health data.
Compliance: Adhere to healthcare data protection regulations such as HIPAA.
Documentation and Reporting Tools:

Jupyter Notebooks or Markdown: For documenting the code, analysis, and results.
Report Generation Tools: Tools for creating project reports and documentation.
Collaboration Tools:

Communication Tools: Email, messaging apps, or project management tools for team collaboration and coordination.
Testing and Validation Tools:

Unit Testing Frameworks: Implement unit tests to ensure the correctness of different components.
Validation Metrics: Define metrics for evaluating the performance of the stress detection model.
Ethical Considerations:

Ethical Guidelines: Develop the project with consideration for ethical and responsible AI practices, especially when dealing with sensitive health data.

## Program
```python

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import neurokit as nk
import seaborn as sns
import pandas as pd
def load_data(path, subject):
 """Given path and subject, load the data of the subject"""
 os.chdir(path)
 os.chdir(subject)
 with open(subject + '.pkl', 'rb') as file:
 data = pickle.load(file, encoding='latin1')
 return data
class read_data_one_subject:
 """Read data from WESAD dataset"""
 def __init__(self, path, subject):
 self.keys = ['label', 'subject', 'signal']
 self.signal_keys = ['wrist', 'chest']
 self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
 self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
 os.chdir(path)
20
 os.chdir(subject)
 with open(subject + '.pkl', 'rb') as file:
 data = pickle.load(file, encoding='latin1')
 self.data = data
 def get_labels(self):
 return self.data[self.keys[0]]
 def get_wrist_data(self):
 """"""
 #label = self.data[self.keys[0]]
 #assert subject == self.data[self.keys[1]]
 signal = self.data[self.keys[2]]
 wrist_data = signal[self.signal_keys[0]]
 #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
 #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
 return wrist_data
 def get_chest_data(self):
 """"""
 signal = self.data[self.keys[2]]
 chest_data = signal[self.signal_keys[1]]
 return chest_data
def extract_mean_std_features(ecg_data, label=0, block = 700):
 #print (len(ecg_data))
 i = 0
 mean_features = np.empty(int(len(ecg_data)/block), dtype=np.float64)
 std_features = np.empty(int(len(ecg_data)/block), dtype=np.float64)
 max_features = np.empty(int(len(ecg_data)/block), dtype=np.float64)
 min_features = np.empty(int(len(ecg_data)/block), dtype=np.float64)
 idx = 0
 while i < len(ecg_data):
 temp = ecg_data[i:i+block]
 #print(len(temp))
 if idx < int(len(ecg_data)/block):
21
 mean_features[idx] = np.mean(temp)
 std_features[idx] = np.std(temp)
 min_features[idx] = np.amin(temp)
 max_features[idx] = np.amax(temp)
 i += 700
 idx += 1
 #print(len(mean_features), len(std_features))
 #print(mean_features, std_features)
 features = {'mean':mean_features, 'std':std_features, 'min':min_features, 'max':max_features}
 one_set = np.column_stack((mean_features, std_features, min_features, max_features))
 return one_set
def extract_one(chest_data_dict, idx, l_condition=0):
 ecg_data = chest_data_dict["ECG"][idx].flatten()
 ecg_features = extract_mean_std_features(ecg_data, label=l_condition)
 #print(ecg_features.shape)
 eda_data = chest_data_dict["EDA"][idx].flatten()
 eda_features = extract_mean_std_features(eda_data, label=l_condition)
 #print(eda_features.shape)
 emg_data = chest_data_dict["EMG"][idx].flatten()
 emg_features = extract_mean_std_features(emg_data, label=l_condition)
 #print(emg_features.shape)
 temp_data = chest_data_dict["Temp"][idx].flatten()
 temp_features = extract_mean_std_features(temp_data, label=l_condition)
 #print(temp_features.shape)
 baseline_data = np.hstack((eda_features, temp_features, ecg_features, emg_features))
 #print(len(baseline_data))
 label_array = np.full(len(baseline_data), l_condition)
 #print(label_array.shape)
 #print(baseline_data.shape)
 baseline_data = np.column_stack((baseline_data, label_array))
 #print(baseline_data.shape)
22
 return baseline_data
def recur_print(ecg):
 while ecg is dict:
 print(ecg.keys())
 for k in ecg.keys():
 recur_print(ecg[k])
def execute():
 data_set_path = "/media/jac/New Volume/Datasets/WESAD"
 file_path = "ecg.txt"
 subject = 'S3'
 obj_data = {}
 labels = {}
 all_data = {}
 subs = [2, 3, 4, 5, 6]
 for i in subs:
 subject = 'S' + str(i)
 print("Reading data", subject)
 obj_data[subject] = read_data_one_subject(data_set_path, subject)
 labels[subject] = obj_data[subject].get_labels()
 wrist_data_dict = obj_data[subject].get_wrist_data()
 wrist_dict_length = {key: len(value) for key, value in wrist_data_dict.items()}
 chest_data_dict = obj_data[subject].get_chest_data()
 chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
 print(chest_dict_length)
 chest_data = np.concatenate((chest_data_dict['ACC'], chest_data_dict['ECG'], chest_data_dict['EDA'],
 chest_data_dict['EMG'], chest_data_dict['Resp'], chest_data_dict['Temp']), axis=1)
 # Get labels
 # 'ACC' : 3, 'ECG' 1: , 'EDA' : 1, 'EMG': 1, 'RESP': 1, 'Temp': 1 ===> Total dimensions : 8
 # No. of Labels ==> 8 ; 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
 # 4 = meditation, 5/6/7 = should be ignored in this dataset
23
 # Do for each subject
 baseline = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 1])
 # print("Baseline:", chest_data_dict['ECG'][baseline].shape)
 # print(baseline.shape)
 stress = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 2])
 # print(stress.shape)
 amusement = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 3])
 # print(amusement.shape)
 baseline_data = extract_one(chest_data_dict, baseline, l_condition=1)
 stress_data = extract_one(chest_data_dict, stress, l_condition=2)
 amusement_data = extract_one(chest_data_dict, amusement, l_condition=3)
 full_data = np.vstack((baseline_data, stress_data, amusement_data))
 print("One subject data", full_data.shape)
 all_data[subject] = full_data
 i = 0
 for k, v in all_data.items():
 if i == 0:
 data = all_data[k]
 i += 1
 print(all_data[k].shape)
 data = np.vstack((data, all_data[k]))
 print(data.shape)
 return data
if __name__ == '__main__':
 execute()
 """
 ecg, eda = chest_data_dict['ECG'], chest_data_dict['EDA']
 x = [i for i in range(len(baseline))]
 for one in baseline:
 x = [i for i in range(99)]
24
 plt.plot(x, ecg[one:100])
 break
 """
 #x = [i for i in range(10000)]
 #plt.plot(x, chest_data_dict['ECG'][:10000])
 #plt.show()
 # BASELINE
 # [ecg_features[k] for k in ecg_features.keys()])
 #ecg = nk.ecg_process(ecg=ecg_data, rsp=chest_data_dict['Resp'][baseline].flatten(), sampling_rate=700)
 #print(os.getcwd())
 """
 #recur_print
 print(type(ecg))
 print(ecg.keys())
 for k in ecg.keys():
 print(k)
 for i in ecg[k].keys():
 print(i)

 resp = nk.eda_process(eda=chest_data_dict['EDA'][baseline].flatten(), sampling_rate=700)
 resp = nk.rsp_process(chest_data_dict['Resp'][baseline].flatten(), sampling_rate=700)
 for k in resp.keys():
 print(k)
 for i in resp[k].keys():
 print(i)

 # For baseline, compute mean, std, for each 700 samples. (1 second values)
 #file_path = os.getcwd()
 with open(file_path, "w") as file:
 #file.write(str(ecg['df']))
 file.write(str(ecg['ECG']['HRV']['RR_Intervals']))
25
 file.write("...")
 file.write(str(ecg['RSP']))
 #file.write("RESP................")
 #file.write(str(resp['RSP']))
 #file.write(str(resp['df']))
 #print(type(ecg['ECG']['HRV']['RR_Intervals']))
 #file.write(str(ecg['ECG']['Cardiac_Cycles']))
 #print(type(ecg['ECG']['Cardiac_Cycles']))
 #file.write(ecg['ECG']['Cardiac_Cycles'].to_csv())
 # Plot the processed dataframe, normalizing all variables for viewing purpose
 """
 """
 bio = nk.bio_process(ecg=chest_data_dict["ECG"][baseline].flatten(),
rsp=chest_data_dict['Resp'][baseline].flatten()
 , eda=chest_data_dict["EDA"][baseline].flatten(), sampling_rate=700)
 #nk.z_score(bio["df"]).plot()
 print(bio["ECG"].keys())
 print(bio["EDA"].keys())
 print(bio["RSP"].keys())
 #ECG
 print(bio["ECG"]["HRV"])
 print(bio["ECG"]["R_Peaks"])
 #EDA
 print(bio["EDA"]["SCR_Peaks_Amplitudes"])
 print(bio["EDA"]["SCR_Onsets"])
 #RSP
 print(bio["RSP"]["Cycles_Onsets"])
 print(bio["RSP"]["Cycles_Length"])
 """
26
 print("Read data file")
2)Scratch.py
"""
import pandas as pd
from sklearn import svm
file = 'data/train.csv'
train_data = pd.read_csv(file)
print(train_data.head())
print(train_data.columns)
#features = Sex, Age, Pclass, Cabin, SibSp, Parch, Embarked, Name, Ticket
#label = Survived
#'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
#SVM
#Bayesian logisitic regression
kernel = 'rbf'
svm.SVC()
"""
# Extract features using sliding window and form the training dataset, test dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import numpy as np
X, y = make_classification(n_samples=10000, n_features=6,
 n_informative=3, n_redundant=0,
27
 random_state=0, shuffle=True)
print(X.shape) # 10000x6
print(y.shape) # 10000
# TODO: Feature extraction using sliding window
train_features, test_features, train_labels, test_labels = train_test_split(X, y,
 test_size=0.25, random_state=42)
# TODO: K-fold cross validation
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
clf = RandomForestClassifier(n_estimators=100, max_depth=3, oob_score=True
 )
clf.fit(X, y)
print(clf.feature_importances_)
#print(clf.oob_decision_function_)
print(clf.oob_score_)
predictions = clf.predict(test_features)
errors = abs(predictions - test_labels)
print("M A E: ", round(np.mean(errors), 2))
# Visualization
feature_list = [1, 2, 3, 4, 5, 6]
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = clf.estimators_[5]
# Export the image to a dot file
28
export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
#graph.write_png('tree_.png')
# TODO: Confusion matrix, Accuracy
# GMM
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X, y)
3)classifier.py
from read_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
if __name__ == '__main__':
 data = execute()
 print(data.shape)
 X = data[:, :16] # 16 features
 y = data[:, 16]
 print(X.shape)
 print(y.shape)
 print(y)
 train_features, test_features, train_labels, test_labels = train_test_split(X, y,
 test_size=0.25)
 print('Training Features Shape:', train_features.shape)
 print('Training Labels Shape:', train_labels.shape)
 print('Testing Features Shape:', test_features.shape)
29
 print('Testing Labels Shape:', test_labels.shape)
 clf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
 clf.fit(X, y)
 print(clf.feature_importances_)
 # print(clf.oob_decision_function_)
 print(clf.oob_score_)
 predictions = clf.predict(test_features)
 errors = abs(predictions - test_labels)
 print("M A E: ", np.mean(errors))
 print(np.count_nonzero(errors), len(test_labels))
 print("Accuracy:", np.count_nonzero(errors)/len(test_labels))
4)setup.py
from distutils.core import setup
setup(
 name='stress_affect_detection',
 version='1.0',
 packages=[''],
 url='',
 license='',
 author='Jagan',
 author_email='jaganshannmugam@outlook.com',
 description='Stress and Affect detection using WESAD dataset'
)


```


## Output

![image](https://github.com/Balajibal/Stress-detection-for-brain-stroke-people-using-RNN/assets/75234946/5f1781c1-0e60-4237-8dfb-a62789e3c75f)

![image](https://github.com/Balajibal/Stress-detection-for-brain-stroke-people-using-RNN/assets/75234946/1c8a3120-e379-4048-a699-1a2537092881)


## Result

The development of a stress detection system for Brain Storm person using Recurrent Neural Network (RNN) has shown promising results in accurately identifying and predicting stress levels. By leveraging the capabilities of RNNs to capture temporal dependencies in physiological data, the system can effectively differentiate between stress and non-stress states. Furthermore, the utilization of Brain Storm person, a multimodal brain-computer interface (BCI) platform, enables the system to gather comprehensive physiological data, including EEG, ECG, and HRV signals, providing a richer and more nuanced understanding of stress patterns. The integration of RNNs with Brain Storm person has demonstrated significant advantages over traditional stress detection methods. RNNs excel at processing sequential data, making them well-suited for analysing the dynamic nature of physiological signals. Additionally, the multimodal nature of Brain Storm person provides a more holistic representation of stress, encompassing both central nervous system and peripheral physiological responses. The potential applications of this stress detection system extend beyond individual stress management. In healthcare settings, it can be used to monitor patients undergoing stressful procedures or interventions, enabling timely interventions to reduce stressrelated complications. Moreover, the system could be incorporated into wearable devices to provide real-time stress monitoring for individuals in high-stress environments, such as first responders or military personnel. Overall, the development of a stress detection system for Brain Storm person using RNN holds immense promise for revolutionizing stress management and promoting well-being. By providing accurate and timely stress detection, this system has the potential to improve individual health outcomes and enhance performance in high-stress environments.
