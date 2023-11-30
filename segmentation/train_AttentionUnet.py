#!/usr/bin/env python
# coding: utf-8

# # Build the model

# In[21]:
import numpy

from models.AttentionUnet import AttU_Net, U_Net, Bi_U_Net
import torch
import torch.nn as nn

model = AttU_Net(num_classes=2, checkpoint=False).cuda()
# model = U_Net(num_classes=2, checkpoint=False).cuda()

# # Create the dataset loader

# In[22]:


import os
from torch.utils.data import DataLoader
from dataset.data_config.Camvid import CamVidDataset, Bi_Dataset

# data_path

title = ""
DATA_DIR = r'C:\Users\SseakomSui\Desktop\wisc\labelme'

'''
Post
'''
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_val_dir = os.path.join(DATA_DIR, 'test')
y_val_dir = os.path.join(DATA_DIR, 'test_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')

train_dataset = CamVidDataset(
    x_train_dir,
    y_train_dir,
)

val_dataset = CamVidDataset(
    x_val_dir,
    y_val_dir,
)

test_dataset = CamVidDataset(
    x_test_dir,
    y_test_dir,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

'''
Bi-temporal

pre_train_dir = os.path.join(DATA_DIR, r'bi_temp\train\pre')
post_train_dir = os.path.join(DATA_DIR, r'bi_temp\train\post')
y_train_dir = os.path.join(DATA_DIR, r'bi_temp\train_labels')

pre_val_dir = os.path.join(DATA_DIR, r'bi_temp\test\pre')
post_val_dir = os.path.join(DATA_DIR, r'bi_temp\test\post')
y_val_dir = os.path.join(DATA_DIR, r'bi_temp\test_labels')

pre_test_dir = os.path.join(DATA_DIR, r'bi_temp\test\pre')
post_test_dir = os.path.join(DATA_DIR, r'bi_temp\test\post')
y_test_dir = os.path.join(DATA_DIR, r'bi_temp\test_labels')

train_dataset = Bi_Dataset(
    pre_train_dir,
    post_train_dir,
    y_train_dir
)

val_dataset = Bi_Dataset(
    pre_val_dir,
    post_val_dir,
    y_val_dir
)

test_dataset = Bi_Dataset(
    pre_test_dir,
    post_test_dir,
    y_test_dir
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
'''
# # train



from d2l import torch as d2l
import pandas as pd
import numpy as np
import os
from utils.train_utils import evaluate_loss, combined_loss

epochs_num = 34

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.5)

device = torch.device("cuda:0")
# loss_func = nn.CrossEntropyLoss(ignore_index=255)
'''
dice loss
'''
import torch.nn.functional as F


def dice_loss(y_pred, y_true, smooth=1):
    y_pred = F.softmax(y_pred, dim=1)  # apply softmax to y_pred along the channel dimension
    y_pred = y_pred[:, 1, :, :]  # take the probability of the foreground class
    intersection = (y_pred * y_true).sum()
    sum_masks = y_pred.sum() + y_true.sum()
    dice = (2 * intersection + smooth) / (sum_masks + smooth)
    return 1 - dice

def focal_loss(y_pred, y_true, alpha=0.25, gamma=2, smooth=1):
    y_pred = F.softmax(y_pred, dim=1)  # apply softmax to y_pred along the channel dimension
    y_pred = y_pred[:, 1, :, :]  # take the probability of the foreground class
    intersection = (y_pred * y_true).sum()
    sum_masks = y_pred.sum() + y_true.sum()
    dice = (2 * intersection + smooth) / (sum_masks + smooth)
    focal_dice = (1 - dice) ** gamma
    focal_loss = focal_dice * alpha
    return focal_loss

def combine_loss(y_pred, y_true):
    bce_loss = F.cross_entropy(y_pred, y_true)

    # Resize y_true to match the size of y_pred
    y_true_resized = F.interpolate(y_true.unsqueeze(1), size=y_pred.shape[2:], mode='nearest').squeeze(1)

    dl_loss = dice_loss(y_pred, y_true_resized)
    return bce_loss + dl_loss


loss_func = combine_loss

'''
post train
'''


def train_model(model, train_iter, val_iter, loss_func, optimizer, num_epochs, schedule, device=device):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'val loss', 'val acc'])
    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    epochs_list = []
    time_list = []
    lr_list = []

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (X, labels) in enumerate(train_iter):
            timer.start()

            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            gt = labels.long().to(device)

            model.train()
            optimizer.zero_grad()
            result = model(X)
            loss_sum = loss_func(result, gt)
            loss_sum.sum().backward()
            optimizer.step()
            acc = d2l.accuracy(result, gt)
            metric.add(loss_sum, acc, labels.shape[0], labels.numel())

            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None, None))

        schedule.step()
        val_acc = d2l.evaluate_accuracy_gpu(model, val_iter, device=device)
        val_loss = evaluate_loss(model, val_iter, loss_func, device=device)

        animator.add(epoch + 1, (None, None, val_loss, val_acc))
        print(
            f"epoch {epoch + 1}/{epochs_num} --- train loss {metric[0] / metric[2]:.3f} --- train acc {metric[1] / metric[3]:.3f} --- val loss {val_loss:.3f} --- val acc {val_acc:.3f} --- lr {optimizer.state_dict()['param_groups'][0]['lr']} --- cost time {timer.sum()}")


        df = pd.DataFrame()
        train_loss_list.append(metric[0] / metric[2])
        val_loss_list.append(val_loss)

        train_acc_list.append(metric[1] / metric[3])
        val_acc_list.append(val_acc)
        epochs_list.append(epoch + 1)
        time_list.append(timer.sum())
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        df['epoch'] = epochs_list
        df['train_loss'] = train_loss_list
        df['val_loss'] = val_loss_list
        df['train_acc'] = train_acc_list
        df['val_acc'] = val_acc_list
        df["lr"] = lr_list
        df['time'] = time_list

        df.to_excel("trash/Unet_OnlyPost_small.xlsx")

        if np.mod(epoch + 1, 5) == 0:
            torch.save(model.state_dict(), f'trash/TransUnet_{epoch + 1}.pth')

    torch.save(model.state_dict(), f'trash/Att.pth')


train_model(model, train_loader, val_loader, loss_func, optimizer, epochs_num, schedule, device)

# # Test

# In[24]:


from utils.train_utils import test_gpu, show_result_pyplot, inference_model
from sklearn.metrics import confusion_matrix

seg_pred, acc = test_gpu(model, test_loader, device=torch.device("cuda:0"))
print(acc)
# show_result_pyplot(model, img_path, result = seg_pred[0])

# get confusion matrix
y_true = []
y_pred = []
for data, labels in test_loader:
    data, labels = data.to(device), labels.to(device)
    output = model(data)
    _, predicted = torch.max(output.data, 1)
    y_true += labels.view(-1).cpu().numpy().tolist()
    y_pred += predicted.view(-1).cpu().numpy().tolist()

conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)

# # Inference

# In[25]:


from utils.train_utils import inference_model, show_result_pyplot


img_path = r"C:\Users\SseakomSui\Desktop\origin\segmentation\r1\250.png"
# save_path = r"C:\Users\SseakomSui\Desktop\spain\u.png"
result = inference_model(model, img_path)
show_result_pyplot(model, img_path, result)

# Calculate AUC and AUPR
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

y_pred = model(data).detach().cpu().numpy()
y_true = labels.detach().cpu().numpy()



import numpy as np


arr1, arr2 = np.split(y_pred, 2, axis=1)

# Squeeze arr2 to shape (1, 224, 224)
arr2_squeezed = np.squeeze(arr2, axis=1)


arr_r = np.add(arr1, arr2_squeezed)


arr_r_norm = arr_r / np.linalg.norm(arr_r)

fpr, tpr, _ = roc_curve(y_true.ravel(), arr_r_norm.ravel())
roc_auc = auc(fpr, tpr)

data_RoC = {'FPR': fpr, 'TPR': tpr}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data_RoC)

# Save the DataFrame to an Excel file with headers
df.to_excel('RoC5.xlsx', index=False)

precision, recall, _ = precision_recall_curve(y_true.ravel(), arr_r_norm.ravel())
pr_auc = auc(recall, precision)

# Plot ROC curve

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot precision-recall curve
plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()







