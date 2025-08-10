import numpy as np
import matplotlib.pyplot as plt

N = 2
H = 3
T = 20

dh = np.ones((N, H))
np.random.seed(0)
# Wh = np.random.randn(H, H)  # 폭발
Wh = np.random.randn(H, H) * 0.3 # 소실

norm_list = []
for t in range(T):
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dh ** 2)) / N
    norm_list.append(norm)

plt.plot(norm_list)
plt.xlabel('Time Step')
plt.ylabel('Gradient Norm')
plt.title('RNN Gradient Norm')
plt.show()