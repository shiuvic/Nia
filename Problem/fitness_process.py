import numpy as np
import matplotlib.pyplot as plt
import os
nowpath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(nowpath)
parentDirectory = os.path.dirname(fileDirectory)
data_path = os.path.join(parentDirectory, 'data')


def save(reward, name, times, type):
    x = reward
    target_path = os.path.join(data_path, name)
    fit = np.load(os.path.join(target_path, '%s_fit_val(%d).npy' % (name, times)))

    if fit.size == 0:
        fit = np.append(fit, x)
    else:
        last = fit[-1]
        if type:
            if x > last:
                fit = np.append(fit, x)
            else:
                fit = np.append(fit, last)
        else:
            if x < last:
                fit = np.append(fit, x)
            else:
                fit = np.append(fit, last)
    np.save(os.path.join(target_path, '%s_fit_val(%d).npy' % (name, times)), fit)


def clear(name, times):
    a = []
    for i in name:
        target_path = os.path.join(data_path, i)
        if not os.path.isdir(target_path):
            target_path = os.path.join(data_path, i)
            os.mkdir(target_path)
        np.save(os.path.join(target_path, '%s_fit_val(%d).npy' % (i, times)), a)


def show(name, times):
    for i in name:
        target_path = os.path.join(data_path, i)
        data1 = np.load(os.path.join(target_path, '%s_fit_val(%d).npy' % (i, times)))
        plt.plot(data1, label=i)
    plt.title('NiaPy Learning Curve')
    plt.legend()
    plt.show()