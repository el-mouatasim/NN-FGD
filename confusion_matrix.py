#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:

    # Calculate Confusion Matrix
    @staticmethod
    def calculate(y, yhat):
        return confusion_matrix(y, yhat)

    # Plot Confusion Matrix
    @staticmethod
    def plot(title, confusion, matrix_dimension):
        plt.clf()
        plt.xlabel('Actual Target (y)')
        plt.ylabel('Predicted Target (yhat)')
        plt.grid(False)
        plt.xticks(np.arange(matrix_dimension))
        plt.yticks(np.arange(matrix_dimension))
        plt.title(title)
        plt.imshow(confusion, cmap=plt.cm.jet, interpolation='nearest')

        for i, cas in enumerate(confusion):
            for j, count in enumerate(cas):
                if count > 0:
                    xoff = .07 * len(str(count))
                    plt.text(j - xoff, i + .2, int(count), fontsize=9, color='white')

        #plt.savefig(r'confusion_matrices/' + str(title))
        plt.savefig("confusion_matrices.png",fmt="png")
        #fig.savefig("../plots/simple_linear/cost_vs_iterations_plot_comp.png", fmt="png")
