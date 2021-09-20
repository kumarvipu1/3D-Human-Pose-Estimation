import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import datetime

def animate3d(predictions):
    now = datetime.datetime.now()
    POSE_PAIRS = [[1, 0], [1, 2], [1, 3], [1, 6], [3, 4], [4, 5],
                  [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14]]

    keypoint_list = predictions
    grid_size = int(len(keypoint_list[0]) / 2) + 1
    print(grid_size)
    fig = plt.figure(figsize=(15, 15))
    for i in range(len(keypoint_list)):
        print('writing for frame {} of frames {}....'.format(i, len(keypoint_list)))
        kparray = keypoint_list[i]
        axs = fig.add_subplot(111, projection='3d')
        axs.view_init(10, -75)
        axs.set_xlim3d(250, 850)
        axs.set_zlim3d(-500, 0)
        axs.set_ylim3d(0, 60)
        for j in range(len(kparray)):
            kpts = np.array(kparray[j])
            x = kpts[..., 0]
            y = kpts[..., 2]
            z = -kpts[..., 1]
            for k in POSE_PAIRS:
                try:
                    axs.plot((x[k[0]], x[k[1]]), (y[k[0]], y[k[1]]), (z[k[0]], z[k[1]]))
                    axs.scatter(x, y, z, s=4)
                    plt.pause(0.0001)
                except:
                    continue
        savepath = os.path.join('output3d/figures', '{}.png'.format(i))
        fig.savefig(savepath)
    images = []
    for files in os.listdir('output3d/figures'):
        path = os.path.join('output3d/figures', files)
        images.append(imageio.imread(path))
    imageio.mimsave('output3d/movie_{}_{}_{}.gif'.format(str(now.month),str(now.day),str(now.year)), images)