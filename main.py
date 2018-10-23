#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
data=pd.read_csv('creditcard.csv')


# In[63]:


data.tail()


# In[64]:


data.describe()


# In[65]:


numcols = data.columns
numcols


# In[ ]:





# In[66]:


Features = np.array(data[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
Features[2, :]


# In[67]:


import numpy.random as nr
import sklearn.model_selection as ms
nr.seed(1234)
labels = np.array(data['Class'])
index = range(data.shape[0])
index = ms.train_test_split(index,test_size=0.3)
x_train = np.array(Features[index[0], :])
y_train = np.ravel(labels[index[0]])
x_test = np.array(Features[index[1], :])
y_test = np.ravel(labels[index[1]])
print (x_train.shape)
print (y_train.shape)


# In[68]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(fit_intercept = False)
log_model.fit(x_train, y_train)


# In[69]:


print(log_model.intercept_)
print(log_model.coef_)


# In[72]:


import sklearn.metrics as sklm
y_hat_p = log_model.predict_proba(x_test)
def score_model(y_p, threshold):
    return np.array([1 if x > threshold else 0 for x in y_p[:, 1]])
y_hat = score_model(y_hat_p, 0.9)

def print_metrics(y_true, y_predicted):
    metrics = sklm.precision_recall_fscore_support(y_true, y_predicted)
    cfmat = sklm.confusion_matrix(y_true, y_predicted)
    print("                     Predicted Positive        Predicted Negative")
    print("Actually Positive    %6d" %cfmat[0][0] + "                  %6d" %cfmat[0][1])
    print("Actually Negative    %6d" %cfmat[1][0] + "                  %6d" %cfmat[1][1])
    print("")
    print("Accuracy: " + str(sklm.accuracy_score(y_true, y_predicted)))
    print("")
    print("            Positive       Negative")
    print("Num Cases:  %6f"%metrics[3][0] + "          %6.2f"%metrics[3][1]) 
    print("precision:  %6.2f"%metrics[0][0] + "          %6.2f"%metrics[0][1])
    print("Recall:     %6.2f"%metrics[1][0] + "          %6.2f"%metrics[1][1])
    print("fscore:     %6.2f"%metrics[2][0] + "          %6.2f"%metrics[2][1])
    
print_metrics(y_test, y_hat)


# In[71]:


def plot_roc(y_true, prob):
    fpr, tpr, threshold = sklm.roc_curve(y_true, prob[:, 1])
    auc = sklm.auc(fpr, tpr)
    plt.plot(fpr, tpr, color = "orange", label = 'auc %0.2f' % auc)
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylabel([0,1])
    plt.title("Reciever Operating Characterstics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc = 'lower right')
    plt.show()
plot_roc(y_test, y_hat_p)


# In[ ]:




