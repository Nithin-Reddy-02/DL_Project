import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

path_x = sys.argv[1]
path_y = sys.argv[2]
path_z = sys.argv[3]

def average_out_embeddings(npy_file_path):
    x=np.load(npy_file_path,allow_pickle='true')
    length = len(x)
    x_split_data=x[:,0]
    y_split=x[:,1]
    y_split=y_split.astype('int')
    x_split=np.zeros([length,200])
    print(len(x_split_data))
    print(len(x_split_data[1]))
    # print(x_split_data[0])
    for i in range(0,length):
        # len(x_split_data[i])
        b=np.zeros([len(x_split_data[i]),200])
        for j in range(0,len(x_split_data[i])):
            b[j,:]=x_split_data[i][j]
        b=np.sum(b,axis=0)
        x_split[i,:]=b/len(x_split_data[i])
    return x_split, y_split
x_train, y_train = average_out_embeddings(path_x)
x_test, y_test = average_out_embeddings(path_y)
x_dev, y_dev = average_out_embeddings(path_z)

y_train1=np.zeros([5082,1])
for i in range(0,5082):
    y_train1[i][0]=y_train[i]
y_test1=np.zeros([1517,1])
for i in range(0,1517):
    y_test1[i][0]=y_test[i]
y_dev1=np.zeros([994,1])
for i in range(0,994):
    y_dev1[i][0]=y_dev[i]


def metrics_calculator(preds, test_labels):
    cm = confusion_matrix(test_labels, preds)
    print((cm[0][0]-40)/(cm[0][0]+cm[0][1]))
    print(cm[0][0]/(cm[0][0]+cm[1][0]))
    return ((cm[0][0])/(cm[0][0]+cm[0][1])), (cm[0][0]/(cm[0][0]+cm[1][0]))

def LR_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels):
    f = open("LR_results.txt", "w+")
    try:
        print(len(train_avg))
        print(len(train_avg[0]))
        print(train_avg[0])
        LR = LogisticRegression(C=1)
        LR.fit(train_avg, train_labels)
        d_preds = LR.predict(dev_avg)
        d_res = "Accuracy -> " + str((accuracy_score(d_preds, dev_labels)*100)) + "\n"
        precision,recall = metrics_calculator(d_preds, dev_labels)
        d_metrics = " Precision: " + str(precision) + " Recall: " + str(recall) + " F1-score: " + str((2*precision*recall)/(precision+recall))  + "\n"
        f.write("Dev set:\n"+ d_res + d_metrics)
            
        t_preds = LR.predict(test_avg)
        t_res = "Accuracy -> " + str((accuracy_score(t_preds, test_labels)*100)) + "\n"
        precision,recall = metrics_calculator(t_preds, test_labels)
        t_metrics = " Precision: " + str(precision) + " Recall: " + str(recall) + " F1-score: " + str((2*precision*recall)/(precision+recall))  + "\n"
        f.write("Test set:\n"+ t_res + t_metrics + "\n\n")
    except:
        f.close()
        
    f.close()


LR_scores(x_train,x_dev,x_test,y_train1,y_dev1,y_test1)
