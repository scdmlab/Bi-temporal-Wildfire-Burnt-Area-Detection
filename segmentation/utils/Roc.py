import matplotlib.pyplot as plt
import numpy as np

# 定义focal loss函数
def focal_loss(p, gamma):
    return - (1 - p) ** gamma * np.log(p)

# 真实类别的概率
p = np.linspace(0.01, 1, 100)

# γ的取值
gammas = [0, 0.5, 1, 2, 5]

# 颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange']

for gamma, color in zip(gammas, colors):
    loss = focal_loss(p, gamma)
    if gamma == 0:
        plt.plot(p, loss, color=color, label='γ=0(Cross-Entropy)')
    else:
        plt.plot(p, loss, color=color, label=f'γ={gamma}')

# 添加图例
plt.legend()

# 设置坐标轴标签
plt.xlabel('Probability of Ground Truth Class')
plt.ylabel('Loss')

# 在γ=0的部分标注上Cross-Entropy

# 显示图形
plt.show()
