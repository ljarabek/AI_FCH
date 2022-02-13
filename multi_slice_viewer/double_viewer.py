import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def process_key(event):
    fig = event.canvas.figure
    ax1, ax2, ax3 = fig.axes[0], fig.axes[1], fig.axes[2]
    if event.key == 'j':
        previous_slice(ax1, ax2, ax3)
    elif event.key == 'k':
        next_slice(ax1, ax2, ax3)
    fig.canvas.draw()


def previous_slice(ax1, ax2, ax3):
    v1 = ax1.volume
    s1 = ax1.segmentation
    ax1.index = (ax1.index - 1) % v1.shape[0]
    ax1.images[0].set_array(v1[ax1.index])
    ax1.images[1].set_array(s1[ax1.index])

    v2 = ax2.volume
    s2 = ax2.segmentation
    ax2.index = (ax2.index - 1) % v2.shape[0]
    ax2.images[0].set_array(v2[ax2.index])
    ax2.images[1].set_array(s2[ax2.index])

    v3 = ax3.volume
    s3 = ax3.segmentation
    ax3.index = (ax3.index - 1) % v3.shape[0]
    ax3.images[0].set_array(v3[ax3.index])
    ax3.images[1].set_array(s3[ax3.index])
    # ax.title(ax.index)


def next_slice(ax1, ax2, ax3):
    # ax1, ax2, ax3 = ax
    v1 = ax1.volume
    s1 = ax1.segmentation
    ax1.index = (ax1.index + 1) % v1.shape[0]
    ax1.images[0].set_array(v1[ax1.index])
    ax1.images[1].set_array(s1[ax1.index])

    v2 = ax2.volume
    s2 = ax2.segmentation
    ax2.index = (ax2.index + 1) % v2.shape[0]
    ax2.images[0].set_array(v2[ax2.index])
    ax2.images[1].set_array(s2[ax2.index])

    v3 = ax3.volume
    s3 = ax3.segmentation
    ax3.index = (ax3.index + 1) % v3.shape[0]
    ax3.images[0].set_array(v3[ax3.index])
    ax3.images[1].set_array(s3[ax3.index])
    # ax.title(ax.index)


def seg_viewer(v1, s1, v2, s2, v3, s3, cmap1="jet", cmap2='Reds', cmap3='jet'):
    remove_keymap_conflicts({'j', 'k'})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.volume = v1
    ax1.segmentation = s1
    ax2.volume = v2
    ax2.segmentation = s2
    ax3.volume = v3
    ax3.segmentation = s3

    ax1.index = v1.shape[0] // 2
    ax1.imshow(v1[ax1.index], cmap='gray')
    ax1.imshow(s1[ax1.index], cmap=cmap1, alpha=0.2)

    ax2.index = v2.shape[0] // 2
    ax2.imshow(v2[ax2.index], cmap='gray')
    ax2.imshow(s2[ax2.index], cmap=cmap2, alpha=0.2)

    ax3.index = v3.shape[0] // 2
    ax3.imshow(v3[ax3.index], cmap='gray')
    ax3.imshow(s3[ax3.index], cmap=cmap3, alpha=0.2)

    # ax.title = str(ax.index)
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


# prikaže 3D numpy array z večimi slici/odseki slike
def display_numpy(picture):
    fig = plt.figure()
    iter = int(len(picture) / 30)
    for num, slice in enumerate(picture):
        if num >= 30:
            break
        y = fig.add_subplot(5, 6, num + 1)

        y.imshow(picture[num * iter], cmap='gray')
    plt.show()
    return
