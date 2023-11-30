import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(r"C:\Users\SseakomSui\Desktop\Empowering Wildfire Burnt Area Detection with Deep Learning\histogram matching\kenya\520.png",
                 flags=1)
imgRef = cv2.imread(r"C:\Users\SseakomSui\Desktop\Empowering Wildfire Burnt Area Detection with Deep Learning\histogram matching\ref.png",
                    flags=1)

_, _, channel = img.shape
imgOut = np.zeros_like(img)
for i in range(channel):
    print(i)
    histImg, _ = np.histogram(img[:, :, i], 256)
    histRef, _ = np.histogram(imgRef[:, :, i], 256)
    cdfImg = np.cumsum(histImg)
    cdfRef = np.cumsum(histRef)
    for j in range(256):
        tmp = abs(cdfImg[j] - cdfRef)
        tmp = tmp.tolist()
        index = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
        imgOut[:, :, i][img[:, :, i] == j] = index

fig = plt.figure(figsize=(10, 7))
plt.subplot(231), plt.title("Original image"), plt.axis('off')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(232), plt.title("Matching template"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgRef, cv2.COLOR_BGR2RGB))
plt.subplot(233), plt.title("Matching output"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgOut, cv2.COLOR_BGR2RGB))
histImg, bins = np.histogram(img.flatten(), 256)
plt.subplot(234, yticks=[]), plt.bar(bins[:-1], histImg)
histRef, bins = np.histogram(imgRef.flatten(), 256)
plt.subplot(235, yticks=[]), plt.bar(bins[:-1], histRef)
histOut, bins = np.histogram(imgOut.flatten(), 256)
plt.subplot(236, yticks=[]), plt.bar(bins[:-1], histOut)
plt.show()
save_path = r"C:\Users\SseakomSui\Desktop\Empowering Wildfire Burnt Area Detection with Deep Learning\histogram matching\kenya\kenya_hm.png"
cv2.imwrite(save_path,imgOut)
