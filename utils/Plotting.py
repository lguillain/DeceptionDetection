import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_loss(loss_train, loss_test, title):
    if not os.path.isdir('Figures'):
        os.makedirs('Figures')

    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.title(title)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('Figures/loss_{}'.format(title))


def plot_accuracy(accuracy_train, accuracy_test, title):
    if not os.path.isdir('Figures'):
        os.makedirs('Figures')

    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('Figures/accuracy_{}'.format(title))

def plot_history(test):
    plt.title(title)
    mean = np.array(test).mean(axis=0)
    std = np.array(test).std(axis=0)

    #plt.fill_between(range(0, len(mean)), mean - std,
     #                    mean + std, alpha=0.1)
    #plt.plot(range(0, len(mean)), mean, 'o-')
    plt.errorbar(range(0, len(mean)), mean, yerr=std)
