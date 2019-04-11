import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

fig = plt.figure()
ax = fig.add_subplot(111)

graph_data_f = os.path.join(curdir, '..', 'tmp', 'loss.txt')

def init():
    with open(graph_data_f, "w") as f:
        f.write('')

def plot(step, loss):
    with open(graph_data_f, "a") as f:
        f.write('%s\t%s\n' % (step, loss))

def animate(i):
    graph_data = np.loadtxt(graph_data_f, skiprows=0, delimiter='\t', unpack=True).transpose()
    steps = graph_data[:,0]
    loss = graph_data[:,1]
    filtered = lowess(loss, steps, is_sorted=True, frac=148.0/len(steps), it=0)
    ax.clear()
    ax.set_xlabel('steps')
    ax.set_ylabel('loss')
    ax.plot(steps, loss, 'r')
    ax.plot(filtered[:,0], filtered[:,1], 'b')

def test_draw():
    ani = animation.FuncAnimation(fig, animate, interval=3000)
    plt.show()

if __name__ == '__main__':
    test_draw()
