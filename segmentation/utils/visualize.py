from skimage import io, transform, color
import numpy as np
import cv2
from PIL import Image


def visual_confusedMatrix(img_gt_path, img_pred_path):
    # Load the two RGB images
    img_gt = cv2.imread(img_gt_path)
    img_pred = cv2.imread(img_pred_path)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)

    # Binarize the images
    average_pixel_value1 = cv2.mean(gray1)
    average_pixel_value2 = cv2.mean(gray2)
    average_pixel_value1 = int(average_pixel_value1[0])
    average_pixel_value2 = int(average_pixel_value2[0])
    _, binary1 = cv2.threshold(gray1, average_pixel_value1, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, average_pixel_value2, 255, cv2.THRESH_BINARY)
    binary2 = cv2.bitwise_not(binary2)

    # Merge the binary images and assign colors
    merged = cv2.merge((binary1, binary2, binary2))
    merged[np.where((merged == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
    merged[np.where((merged == [255, 0, 0]).all(axis=2))] = [0, 0, 255]
    merged[np.where((merged == [0, 0, 255]).all(axis=2))] = [255, 0, 0]
    merged[np.where((merged == [255, 255, 255]).all(axis=2))] = [0, 255, 0]

    # Display the resulting confused matrix image
    cv2.imshow('Confused Matrix', merged)
    cv2.imshow('bi_gt', binary1)
    cv2.imshow('bi_pred', binary2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visual_diff(img_u, img_p):
    # Load the two RGB images
    img_gt = cv2.imread(img_u)
    img_pred = cv2.imread(img_p)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)

    # Binarize the images
    average_pixel_value1 = cv2.mean(gray1)
    average_pixel_value2 = cv2.mean(gray2)
    average_pixel_value1 = int(average_pixel_value1[0])
    average_pixel_value2 = int(average_pixel_value2[0])
    _, binary1 = cv2.threshold(gray1, average_pixel_value1, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, average_pixel_value2, 255, cv2.THRESH_BINARY)
    binary2 = cv2.bitwise_not(binary2)
    binary1 = cv2.bitwise_not(binary1)

    # Merge the binary images and assign colors
    merged = cv2.merge((binary1, binary2, binary2))
    merged[np.where((merged == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
    merged[np.where((merged == [255, 0, 0]).all(axis=2))] = [0, 0, 255]
    merged[np.where((merged == [0, 0, 255]).all(axis=2))] = [255, 0, 0]
    merged[np.where((merged == [255, 255, 255]).all(axis=2))] = [0, 255, 0]

    # Display the resulting confused matrix image
    cv2.imshow('diff', merged)
    cv2.imshow('unet', binary1)
    cv2.imshow('proposed-method', binary2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def simple_bi(img_path, x):
    img = Image.open(img_path)
    gray_img = img.convert('L')
    gray_arr = np.array(gray_img)
    threshold = x
    bin_arr = np.where(gray_arr > threshold, 255, 0)
    bin_img = Image.fromarray(bin_arr.astype(np.uint8))
    bin_img.save(r'C:\Users\SseakomSui\Desktop\origin\segmentation\r5\633_result_b.jpeg')


img_gt_path = r'C:\Users\SseakomSui\Desktop\Empowering Wildfire Burnt Area Detection with Deep Learning\histogram matching\au\633_label.png'
img_pred_path = r'C:\Users\SseakomSui\Desktop\Empowering Wildfire Burnt Area Detection with Deep Learning\histogram matching\au\hmr.png'
visual_confusedMatrix(img_gt_path, img_pred_path)

# simple_bi(r'C:\Users\SseakomSui\Desktop\origin\segmentation\r5\633.png', 80)

# img_u = r'C:\Users\SseakomSui\Desktop\origin\segmentation\r2\28_result_p.jpeg'
# img_p = r'C:\Users\SseakomSui\Desktop\origin\segmentation\r2\si_hm.jpeg'
# visual_diff(img_u, img_p)
