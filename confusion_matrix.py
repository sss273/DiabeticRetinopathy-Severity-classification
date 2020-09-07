import pandas as pd
from sklearn.metrics import confusion_matrix , classification_report, precision_score, recall_score, roc_auc_score, cohen_kappa_score
import numpy as np
from sklearn.preprocessing import label_binarize, LabelBinarizer

CSVFILE='/media/sandeep/New Volume/Angel Billy/aptos2019DR/Testset results.csv'
test_df=pd.read_csv(CSVFILE)

actualValue=test_df['Groundtruth']
predictedValue=test_df['Prediction']

actualValue=actualValue.values
predictedValue=predictedValue.values

cmt=confusion_matrix(actualValue,predictedValue)
print(cmt)

# The True Positives are simply the diagonal elements
TP = np.diag(cmt)
print("\nTP:\n%s" % TP)

# The False Positives are the sum of the respective column, minus the diagonal element (i.e. the TP element
FP = np.sum(cmt, axis=0) - TP
print("\nFP:\n%s" % FP)

# The False Negatives are the sum of the respective row, minus the         diagonal (i.e. TP) element:
FN = np.sum(cmt, axis=1) - TP
print("\nFN:\n%s" % FN)

num_classes = 5
TN = []

for i in range(num_classes):
    temp = np.delete(cmt, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TN.append(sum(sum(temp)))
print("\nTN:\n%s" % TN)

precision = TP/(TP+FP)
recall = TP/(TP+FN)
Specificity = TN/(TN+FN)

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, multi_class="ovo", average="weighted")


print(multiclass_roc_auc_score(actualValue,predictedValue, average="macro"))
print("\nPrecision:\n%s" % precision)

print("\nRecall:\n%s" % recall)
print("\nSpecificity:\n%s" % Specificity)
print("\nClassification final report\n", classification_report(actualValue, predictedValue,labels=[0,1,2,3,4], digits = 5))

print("\n kappa score: %s", cohen_kappa_score(actualValue, predictedValue,labels=[0,1,2,3,4]))

print("\n Precision score: %s", precision_score(actualValue, predictedValue, average='macro'))

print("\n recall score: %s", recall_score(actualValue, predictedValue, average='macro'))

print("\nSpecificity score :%s", sum(Specificity)/len(Specificity))