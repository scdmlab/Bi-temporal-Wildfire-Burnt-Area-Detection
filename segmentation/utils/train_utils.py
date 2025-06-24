import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from d2l import torch as d2l
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import warnings


def evaluate_loss(model, data_iter, loss, device):
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)

        y = y.long().to(device)
        out = model(X)
        loss_sum = loss(out, y).sum()
    return loss_sum.cpu().detach().numpy()


def test_gpu(model, data_iter, device, threshold=0.5):
    seg_pred_all = []
    if isinstance(model, nn.Module):
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
                _, c, _, _ = X[0].shape
            else:
                X = X.to(device)
                _, c, _, _ = X.shape
            y = y.long().to(device)

            seg_logit = model(X)

            metric.add(d2l.accuracy(seg_logit, y), d2l.size(y))

            if c == 1:
                seg_pred = (seg_logit > threshold).to(seg_logit).squeeze(1)
            else:
                seg_pred = seg_logit.argmax(dim=1)
            seg_pred_all.append(seg_pred.cpu().detach().numpy())
    # acc = metric[0] / metric[1]
    return seg_pred.cpu().detach().numpy(), metric[0] / metric[1]


def show_result(model,
                img_path,
                result,
                palette=None,
                opacity=0.5):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img = img.copy()
    seg = result
    # print(type(seg))

    if seg.shape[:2] != img.shape[:2]:
        img = cv2.resize(img, seg.shape[:2])

    if palette is None:
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(
            0, 255, size=(model.num_classes, 3))
        np.random.set_state(state)
    palette = np.array(palette)
    assert palette.shape[0] == model.num_classes
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    # print(color_seg.shape)
    # print((type(color_seg)))
    cv2.imwrite(r'C:\Users\SseakomSui\Desktop\origin\segmentation\RoC\t.jpeg', color_seg)
    img = cv2.resize(img, [w, h])
    return img


def show_result_pyplot(model,
                       img_path,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True,
                       out_file=None):
    img = show_result(model, img_path, result, palette=palette, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
    if out_file is not None:
        cv2.imwrite(img, out_file)


import torchvision.transforms as transforms
from PIL import Image


def inference_model(model, img_path, size=[256, 256], transform=None, out_file=None, device=None):
    # Read image
    img = Image.open(img_path).convert('RGB')
    if transform is None:
        transformer = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transformer(img)
    else:
        img = transform(img)
    model.eval()
    if not device:
        device = next(iter(model.parameters())).device
    with torch.no_grad():
        img = img.view(1, *img.shape).to(device)
        result = model(img).argmax(dim=1)
        return result.squeeze(0).cpu().detach().numpy()


import torch.nn.functional as F


def dice_loss(y_pred, y_true, smooth=1):
    intersection = (y_pred * y_true).sum()
    sum_masks = y_pred.sum() + y_true.sum()
    dice = (2 * intersection + smooth) / (sum_masks + smooth)
    return 1 - dice


def focal_loss(y_pred, y_true, alpha=0.25, gamma=2):
    bce = F.binary_cross_entropy(y_pred, y_true)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    weight = alpha_t * (1 - p_t).pow(gamma)
    fl = weight * bce
    return fl.mean()


def combined_loss(y_pred, y_true):
    fc_loss = focal_loss(y_pred, y_true)
    dl_loss = dice_loss(y_pred, y_true)
    return fc_loss + dl_loss
