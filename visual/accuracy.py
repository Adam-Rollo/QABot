import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

fig = plt.figure()
ax = fig.add_subplot(111)
graph_data_f = os.path.join(curdir, '..', 'tmp', 'accuracy.txt')

def init():
    with open(graph_data_f, "w") as f:
        f.write('')

def plot(step, accuracy):
    with open(graph_data_f, "a") as f:
        f.write('%s\t%s\n' % (step, accuracy))

def animate(i):
    graph_data = np.loadtxt(graph_data_f, skiprows=0, delimiter='\t', unpack=True).transpose()
    steps = graph_data[:,0]
    accuracy = graph_data[:,1]
    ax.clear()
    ax.set_xlabel('steps')
    ax.set_ylabel('accuracy')
    ax.plot(steps, accuracy, 'r')

def test_draw():
    ani = animation.FuncAnimation(fig, animate, interval=3000)
    plt.show()

if __name__ == '__main__':
    test_draw()
