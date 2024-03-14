"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 4D array.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'Input 3D or 4D array. Scroll through the first dimension, click through the last'


class IndexTracker(object):
    def __init__(self, ax, X, cmap='gray', alpha=1):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        self.X = X
        self.slices, rows, cols, ims = X[0].shape
        self.images = X[0].shape[-1]
        self.ind = 0
        self.image = 0
        self.cmap = cmap
        self.alpha = alpha
        self.im = list()
        for ind, x in enumerate(self.X):
            if ind == 0:
                self.im.append(ax.imshow(np.squeeze(x[self.ind, :, :, self.image]),
                                         cmap=cmap[ind], vmin=x.min(), vmax=x.max(),
                                         alpha=alpha[ind], interpolation='none',
                                         aspect='auto'))
            elif x.shape == self.X[0].shape:
                masks = np.unique(x)
                for i in masks[1:]:
                    self.ax.contour(x[self.ind, :, :, self.image] == i, levels=0,
                                    cmap=cmap[ind], linewidths=3)
            else:
                masks = np.rollaxis(x, axis=-1)
                for i, ims in enumerate(masks[1:]):
                    ims = np.expand_dims(ims, axis=-1)
                    self.ax.contour(ims[self.ind, :, :, self.image], levels=[0.1, 0.25, 0.5],
                                    colors=3 * [self.cmap[ind].colors[i + 1]], linewidths=[2, 2, 3],
                                    linestyles=['dotted', 'dashed', 'solid'])

        self.ax.set_ylabel('Slice %s' % self.ind)
        self.ax.set_xlabel('Dynamics %s' % self.image)
        self.update()

    def onscroll(self, event):
        if event.inaxes == self.ax:
            # print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.ax.set_ylabel('Slice %s' % self.ind)
            self.update()

    def onpress(self, event):
        if event.inaxes == self.ax:
            # print("%s %s" % (event.button, event.step))
            if str(event.button) == 'MouseButton.RIGHT':
                self.image = (self.image + 1) % self.images
            if str(event.button) == 'MouseButton.LEFT':
                self.image = (self.image - 1) % self.images
            self.ax.set_xlabel('Dynamics %s' % self.image)
            self.update()

    def update(self):
        # Remove all collections
        while any(self.ax.collections): self.ax.collections[0].remove()
        for i, x in enumerate(self.X):
            if i == 0:
                self.im[i].set_data(x[self.ind, :, :, self.image])

            elif x.shape == self.X[0].shape:
                "This part is suboptimal in terms of speed. One could (should!?) be able to utilize 'set_array' but "
                "does not seem to work, have to replot contours each update. "
                masks = np.unique(x)
                for index in masks[1:]:
                    self.ax.contour(x[self.ind, :, :, self.image] == index, levels=0,
                                    colors=self.cmap[i].colors[round(index)],
                                    linewidths=3, alpha=self.alpha[round(i)])
            else:
                masks = np.rollaxis(x, axis=-1)
                for ind, ims in enumerate(masks[1:]):
                    ims = np.expand_dims(ims, axis=-1)
                    self.ax.contour(ims[self.ind, :, :, self.image], levels=[0.1, 0.25, 0.5],
                                    colors=3 * [self.cmap[i].colors[ind + 1]], linewidths=[2, 2, 3],
                                    alpha=self.alpha[round(i)],
                                    linestyles=['dotted', 'dashed', 'solid'])
        self.im[0].axes.figure.canvas.draw()


def plot3D(X):
    "Takes in several nd_arrays (2,3 or 4D) and creates a scrollable plot with optional overlayed contours"
    "Input: nd_array image. Input can also be a list with each element an array."
    "       The first element is the image and all following are either"
    "            - masks with integer labels 0 = background, 1 = first structure and so on or"
    "            - array with shape [imagedims,n] where n are maps for rendering contours"

    fig, ax = plt.subplots(1, 1)

    if type(X) is not list:
        X = [X]

    for ind, x in enumerate(X):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=-1)
        X[ind] = x

    cmap1 = ListedColormap(['black', 'pink', 'yellow', 'brown'])
    cmap1 = ListedColormap(['black', 'green', 'green', 'green'])
    cmap2 = ListedColormap(['black', 'yellow', 'yellow', 'yellow'])
    cmap3 = ListedColormap(['black', 'red', 'red', 'red'])
    tracker = IndexTracker(ax, X, cmap=['gray', cmap1, cmap2, cmap3], alpha=[1, 1, 1, 1])
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

    fig.canvas.mpl_connect('button_press_event', tracker.onpress)
    # fig.canvas.mpl_connect('scroll_event', tracker.update)
    plt.show()
    return fig, tracker


def plot3D_multiView(X):
    "Same as plot3D but creates plot along remaining axes as well"
    fig, ax = plt.subplots(1, 3, figsize=(20, 3))

    if type(X) is not list:
        X = [X]
    X2 = list()
    X3 = list()
    for ind, x in enumerate(X):
        if x.ndim == 3:
            "Transpose to all directions"
            x2 = np.transpose(x, [1, 0, 2])
            x2 = np.fliplr(x2)
            x3 = np.transpose(x, [2, 0, 1])
            x3 = np.fliplr(x3)
            x = np.expand_dims(x, axis=-1)
            x2 = np.expand_dims(x2, axis=-1)
            x3 = np.expand_dims(x3, axis=-1)
        elif x.ndim == 4:
            x2 = np.transpose(x, [1, 0, 2, 3])
            x2 = np.flip(x2, axis=1)
            x3 = np.transpose(x, [2, 1, 0, 3])
            x3 = np.rot90(x3, axes=(1, 2))
        else:
            raise Exception('Requires 3D or 4D images. If 2D, use plot3D instead')

        X[ind] = x
        X2.append(x3)
        X3.append(x2)

    cmap1 = ListedColormap(['black', 'pink', 'yellow', 'brown'])
    cmap1 = ListedColormap(['black', 'green', 'green', 'green'])
    cmap2 = ListedColormap(['black', 'yellow', 'yellow', 'yellow'])
    cmap3 = ListedColormap(['black', 'red', 'red', 'red'])

    cMap = ['gray', cmap1, cmap2, cmap3]
    alphas = [1, 1, 1, 1]

    tracker = IndexTracker(ax[0], X, cmap=cMap, alpha=alphas)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker.onpress)

    tracker2 = IndexTracker(ax[1], X2, cmap=cMap, alpha=alphas)
    fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker2.onpress)

    tracker3 = IndexTracker(ax[2], X3, cmap=cMap, alpha=alphas)
    fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker3.onpress)

    plt.show()
    return fig, tracker, tracker2, tracker3