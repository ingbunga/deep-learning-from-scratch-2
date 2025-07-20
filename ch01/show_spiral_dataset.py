import sys
sys.path.append('..')
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape)
print('t', t.shape)

markers = ['x', 'o', '^']
for i in range(3):
    mask = t[:,i] == 1
    plt.scatter(x[mask,0], x[mask,1], marker=markers[i])
plt.show()