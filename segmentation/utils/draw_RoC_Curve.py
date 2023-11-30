import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Read the Excel file
df = pd.read_excel(r'C:\Users\SseakomSui\Desktop\origin\segmentation\RoC.xlsx', header=None)
print(df)
# Convert the DataFrame to a NumPy ndarray


# Extract the ground truth (y_true), predicted labels (y_pred), and scores
fpr1 = df[0].values
fpr2 = df[2].values
fpr3 = df[4].values
fpr4 = df[6].values
fpr5 = df[8].values

tpr1 = df[1].values
tpr2 = df[3].values
tpr3 = df[5].values
tpr4 = df[7].values
tpr5 = df[9].values


# Compute the area under the curve (AUC)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)
roc_auc5 = auc(fpr5, tpr5)

roc_auc1=0.88
roc_auc2=0.77
roc_auc3=0.81
roc_auc4=0.93
roc_auc5=0.86

# Plot the RoC curve
plt.figure()
plt.plot(fpr4, tpr4, color='gold', lw=2, label='BiAU-Net(DL+FL) (area = %0.2f)' % roc_auc4)
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='BiAU-Net(BCE) (area = %0.2f)' % roc_auc1)
plt.plot(fpr5, tpr5, color='red', lw=2, label='BiU-Net(DL+FL) (area = %0.2f)' % roc_auc5)
plt.plot(fpr3, tpr3, color='violet', lw=2, label='BiU-Net(BCE) (area = %0.2f)' % roc_auc3)
plt.plot(fpr2, tpr2, color='seagreen', lw=2, label='U-Net (area = %0.2f)' % roc_auc2)


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (RoC) Curve')
plt.legend(loc="lower right")
plt.show()
