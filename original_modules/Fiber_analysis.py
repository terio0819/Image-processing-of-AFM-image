from Image_class import ProcessedImage, Fiber
import imptools
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pprint

plt.rcParams['figure.figsize'] = (3.54, 3.54)
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelpad'] = 4
plt.rcParams['axes.labelsize'] =11
plt.rcParams['xtick.major.pad'] = 6
plt.rcParams['ytick.major.pad'] = 6
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI'] + plt.rcParams['font.sans-serif']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['lines.linewidth'] = 1

plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['ytick.minor.size'] = 2.5

plt.rcParams['lines.markersize'] = 1
plt.rcParams['figure.autolayout'] = True
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True


def track_contour(fiber, save_dir, save_fig):  # todo
    if save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=900)
        ax.plot(fiber.horizon, fiber.height, c='firebrick')
        ax.set_xlim(-50, 900)
        ax.set_ylim(0, 4)
        ax.set_xticks(np.arange(0, 1000, 200))
        ax.set_yticks(np.arange(0, 5, 1))
        for i in fiber.kink_indices:
            ax.axvline(x=fiber.horizon[i],
                       c='red', ls='--', lw=0.75)
        plt.savefig(save_dir, dpi=900)
        plt.close()

def save_fiber_AFM(fiber, save_dir, save_fig):
    if save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=900)
        ax.axis('off')
        ax.imshow(fiber.save_AFM_image, cmap='afmhot')
        ax.scatter(fiber.ytrack[fiber.kink_indices],
                   fiber.xtrack[fiber.kink_indices],
                   c='blue', label='kink')
        plt.savefig(save_dir, dpi=900)
        plt.close()

project_dir = Path('/Users/tomok/Documents/Python/FirstPaperProgram')
data_name_dir = project_dir/'pickle_data/'
result_dir = project_dir/'Figure'


if __name__ == '__main__':
    '''
    save height-length profile and AFM image for each fibers.
    
    '''
    data_name_dict = {n: file.stem for n, file in enumerate(data_name_dir.iterdir()) if file.is_dir()}
    pprint.pprint(data_name_dict)
    data_key = int(input('choose the data to analyse: '))
    fiber_data_dir = data_name_dir / (data_name_dict[data_key]+'/Fiber_data')

    while True:
        ask_save_fig = input('Do you save figures? [y/n]')
        if ask_save_fig == 'y':
            save_fig = True
            break
        elif ask_save_fig == 'n':
            save_fig = False
            break
        else:
            print('enter "y" or "n": ')

    for each_image_dir in fiber_data_dir.iterdir():
        print(each_image_dir.name)
        save_dir = result_dir / data_name_dict[data_key] / 'track' / each_image_dir.name
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        for f in each_image_dir.glob('*.pickle'):
            with open(f, mode='rb') as pickled_file:
                fiber = pickle.load(pickled_file)
                track_contour(fiber, save_dir/f'{f.stem}.svg', save_fig)
                save_fiber_AFM(fiber, save_dir/f'{f.stem}_AFM.png', save_fig)




