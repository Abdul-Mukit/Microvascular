# Introduction
In this project, I demostrate the detection of Diabetic 
Peripheral Neuropathy (DPN) using 5 minute ECG signals. I use Pandas, Scikit-Learn,
Matlab (signal processing & feature extraction) and PyTorch in this project. 
In this repository, I provide codes for:

1. Exporatory Data Analysis using Pandas.
2. Application of traditional machine-learning for classification using Scikit-Learn.
3. Application of deep-learning for classification using PyTorch.


## Dataset
It took me over a year to collect and clean all the data. 
I recorded 24-hour Holter-ECG and SPO2(from both ears) signals of 107 diabetic patients.
I also collected various demographic and medical diagnostic test data. 
The selection criteria were: age 40+, diabetes duration 15+, no previous 
history of hear-attack or stroke.
For creating annotation for Diabetic Peripheral Neuropathy (DPN), 
each of the patients were examined with the Nerve Conduction Velocity test.
This test is the gold standard for detecting DPN.

Among 107 recordings, only 89 were usable due to missing demographic data, 
unusable recordings or due to failure to continue the research. At the 
end, DPN distribution was as below:

* 46 didn't have DPN or were of DPN- class
* 43 had DPN or were of DPN+ class

### 24 Hour Recordings to 5 Minute Recordings
For each of the patients, I have approximately 22-24 hour of Holter ECG 
signal. My objective is to classify a patient using ECG signal of 5 minutes 
length. Thus, I divided each of the Holter ECG signals into small 
recordings of 5 minutes length. 
So I get approximately (43+46)2460/5 = 25632 signals. In reality, I got 
24932 signls of 5 minutes length. Each short signal was labelled 
according to its class - DPN+ or DPN-. Following is the class distribution:
<br/><br/>

| DPN Class  | No. of Patients | No. of Samples / 5min recordings |
|:---:|:---:|:---:|
| Negative (DPN-) | 46 | 13002 |
| Positive (DPN+) | 43 | 11930 |

### HRV Featur Extraction from Each of the 5 Minute Recordings
For using machine learning to classify these signals, I needed to 
extract features from each of these 5m recordings. For this I used 
traditional Heart Rate Variability (HRV) features. In addition, I also 
extracted multi-lag Tone-Entropy features which showed good potential 
in [recent works](https://link.springer.com/article/10.1007/s11517-012-1022-5). 
I acquired 19 HRV features, and 20 Tone-Entropy (from lag 1-10) features.
So, now, for each 5m signal I have 39 features describing it. I did this
part entirely in Matlab.

### ECG to Image Data Conversion for Training CNN
I converted each 5 minute ECG signals into spectrogram images so that 
I can use them to train CNNs. A sample spectrogram image of a 5 minute ECG 
is presented below.

<p align="center">
  <img width="150" height="230" src="spectrogram_sample.png">
</p>



## Codes
Following are the codes and their short summary.

* **DemographEDA**: Exploratory data analysis using Pandas to get a better overview of
the demography and distribution of the dataset.
* **TraditionalClassification**: I use Scikit-Learn to implement binary classification of
DPN. Particularly, I used Decision-trees, Random-forest, Bagging, Boosting and
Support Vector Machines (SVM) to demonstrate their DPN classification capabilities.
In addition, I also show the data pre-processing and feature extraction steps 
that I went through for this classification task.
* **DPNNet_5_2**: Classification using PyTorch based CNN. I used 5 convolutional and 
2 fully connected layers. Thus the surfix "5_2". I started out with my experiemnt
with this network structure.
* **DPNNet_5_3**: CNN netowork with 5 convolutional and 3 fully-connected layers.
The performance improved compared to the "5_2" network.
* **DPNNet_5_3_50pDrop**: Same as the "5_3" network. The only addition is that
I used a 50% dropout for the fully connected layers. Performance Improved
compared to the previous trial.
* **DPNNet_5_3_50pDrop_withBN_withLR**: In addition to the 50% dropout, I
used Batch Normalization after each cov layers and added learning rate 
decay in the training codes. The learning rate begins with 0.01 and every 
100 epochos it is scaled by 0.1. Compared to my previous trials 
the accuracy of the network actually went down this time.
* **TransferLearning**: I implemented Transfer-Learning using ResNet18. 
I froze the entire network but the last fully connected layer. 
The performance did not improve.



## Results
Following are the sensitivity and specificity on the test-data.
The best performing model is the DPNNet_5_3_50pDrop. It achieved 
sensitivity and specificity of 90% and 88% respectively.
<br/><br/>

| Method / Network | Sensitivity(%) | Specificity(%) |
|:---|:---|:---|
| Decision Tree |  73.3  | 48.6 |
| Random Forest | 63.6 | 69.8 |
| Bagging | 70.6 | 54.4 |
| Boosting | 58.7 | 64.0 |
| SVM | 49.7 | 71.6 |
| DPNNet_4_2 | 85 | 88 |
| DPNNet_5_2 | 89 | 86 |
| DPNNet_5_3 | 88 | 88 |
| **DPNNet_5_3_50pDrop \*** | **90 \*** | **88 \*** |
| DPNNet_5_3_50pDrop_withBN_withLR | 85 | 88 |
| TransferLearning | 61 | 62 |


## Current Status of the Research
The research is still being continued at [BIMS](http://bims.uiu.ac.bd/). 
Currently, they are working on several high-impact
journal publications based on this dataset.