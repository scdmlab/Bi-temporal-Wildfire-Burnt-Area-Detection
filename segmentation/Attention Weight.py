import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
y = torch.load("psi_50.pth")# Attention Weight
print(y.shape)
y1=torch.squeeze(y)
print(y1)

import numpy as np
import matplotlib.pyplot as plt

# Example 2D tensor

tensor = y1.detach().numpy()
# Normalize tensor values between 0 and 1
normalized_tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))

# Create a custom color map with red and blue gradient
cmap = plt.cm.get_cmap('viridis')
increased_contrast_tensor = np.power(normalized_tensor, 10)

# Create a figure and plot the normalized tensor using the custom color map
plt.imshow(normalized_tensor, cmap=cmap)

# Add colorbar for reference
plt.colorbar()

# Show the figure
plt.show()
