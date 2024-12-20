# INDIAN LEGAL JUDGMENT PREDICTION USING MULTI-DIMENSIONAL EDGE-EMBEDDED GRAPH CONVOLUTION NETWORK

The job of automatically predicting a court case’s outcome given a text describing the facts of the case is known as The Legal Judgment Prediction (LJP). Most of the previous works on this task were mainly focused on data from foreign settings such as China, Europe, America, etc., where the procedure of law was notably distinct from the Indian Judiciary system. In order to achieve the Indian LJP, a large corpus containing around 35 thousand Indian Supreme Court Cases is collected and further annotated with the corresponding court decisions. Various feature-based models of NLP and other neural networks proposed in recent times have arrived with a lot of shortcomings, such as discarded order and the relation between the words, which resulted in unfavorable results. These models failed to utilize the rich global information from the text. In this thesis work, a Multi-Dimensional Edge-Embedded convolution network is used for the task of LJP. First, a text graph for the full corpus is created to illustrate the undirected and multidimensional link between words and documents. The network is initialized using multi-dimensional, corpus-trained representations of words and documents, and the relations are expressed based on the distance between those nodes. Then, the obtained graph is trained with the GCN, which performs various convolution operations across every dimension. Finally, the prediction of the testing data is done using the obtained GCN to achieve the task of LJP. The developed model outperforms the state-of-the-art methods for the task of Indian LJP.

#Requirements

- numpy
- sklearn
- pandas
- pytorch

# In a Nutshell

Each file in the repo corresponds to each model. Running them with the dataset provided gives us the results

# In Details
```
├──  sent2Vec_LR.py - model corresponding to feature extraction using Sent2Vec and predictions using Logistic Regression
│
│
├──  sent2Vec_RF.py - model corresponding to feature extraction using Sent2Vec and predictions using Random Forest
|
│
├──  sent2Vec_SVM.py - model corresponding to feature extraction using Sent2Vec and predictions using SVMs
|
│
├──  BERT.ipynb - model corresponding to feature extraction using BERT and predictions using our GCN model
|
│
├──  legalBERT.ipynb - model corresponding to feature extraction using legalBERT and predictions using our GCN model
|
│
├──  roBERTa.ipynb - model corresponding to feature extraction using roBERTa and predictions using our GCN model
|
│
├──  roBERTa.ipynb - model corresponding to feature extraction using roBERTa and predictions using our GCN model
|
│
├──  judgment_prediction_dataset.json - dataset with fact description and its label as key value pairs
