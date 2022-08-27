import numpy as np
import matplotlib.pyplot as plt
import os

def save(reward, name, times):
    x = reward
    fit = np.load('/Nia/data/%s/%s_fit_val(%d).npy' % (name, name, times))

    if fit.size == 0:
        fit = np.append(fit, x)
    else:
        last = fit[-1]
        if x < last:
            fit = np.append(fit, x)
        else:
            fit = np.append(fit, last)
    np.save('/Nia/data/%s/%s_fit_val(%d).npy' % (name, name, times), fit)

def clear(name, times):
    a = []
    for i in name:
        if not os.path.isdir('/Nia/data/%s' % i):
            os.mkdir('/Nia/data/%s' % i)
        np.save('/Nia/data/%s/%s_fit_val(%d).npy' % (i, i, times), a)


def show(name, times):
    for i in name:
        data1 = np.load('/Nia/data/%s/%s_fit_val(%d).npy' % (i, i, times))
        plt.plot(data1, label=i)
    plt.title('NiaPy Learning Curve')
    plt.legend()
    plt.show()

