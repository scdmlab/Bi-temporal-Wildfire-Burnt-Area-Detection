# conf_mat=[[440602  20231]
#  [ 84679 859416]]


TP = 0
FN = 0
FP = 0
TN = 0
Accuracy = (TP+TN)/(TP+FN+FP+TN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1=2*Precision*Recall/(Precision+Recall)
TNR=TN/(FP+TN)
BA=(Recall+TNR)/2
Pe=((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/(TP+FN+FP+TN)/(TP+FN+FP+TN)
KC=(Accuracy-Pe)/(1-Pe)
MCC=(TP*TN-FP*FN)/pow((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN),0.5)

print(Accuracy,Precision,Recall,F1,BA,KC,MCC)