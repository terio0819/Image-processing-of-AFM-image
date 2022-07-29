import cv2
from Image_class import ProcessedImage, Fiber
import imptools
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pprint
from typing import List, Union, Dict, TypedDict, Generator, Tuple

plt.rcParams['figure.figsize'] = (3.54, 3.54)
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelpad'] = 4
plt.rcParams['axes.labelsize'] = 11
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


class DentDict(TypedDict):
    kink: float
    ep: float
    straight: float
    kinked_end: float


class BreakdownDict(TypedDict):
    dent: DentDict
    normal: float
    bump: float


def calc_breakdown(image_list: List[ProcessedImage]) -> BreakdownDict:
    out_dict = BreakdownDict(dent=DentDict())
    categories = {'dent', 'normal', 'bump'}
    dent_categories = {'kink', 'ep', 'straight', 'kinked_end'}
    all_length = sum(calc_categ_length(image_list, 'all'))
    for cat in categories:
        if cat == 'dent':
            for d_cat in dent_categories:
                d_cat_component = sum(calc_categ_length(image_list, d_cat)) / all_length
                out_dict['dent'][d_cat] = d_cat_component
        else:
            cat_component = sum(calc_categ_length(image_list, cat)) / all_length
            out_dict[cat] = cat_component

    # todo out_dictの値の合計が1にならない問題があるので、正規化して,さらに合計100になるようにごまかす。
    #  あんまりよくない気もするけど
    _out_dict = out_dict.copy()
    total = 0
    for cat, comp in out_dict.items():
        if cat == 'dent':
            for v in comp.values():
                total += v
        else:
            total += comp

    for cat, comp in out_dict.items():
        if cat == 'dent':
            for dcat, v in comp.items():
                _out_dict['dent'][dcat] = v / total * 100
        else:
            _out_dict[cat] = comp / total * 100
    return _out_dict


def calc_categ_length(image_lst: List[ProcessedImage],
                      category: str) -> Generator[float, None, None]:
    """
    :param image_lst:
    :param category: Category for which you want to generate skeleton image.
                     Category should be set to either 'dent', 'kink','ep','straight','kinked_end', 'normal','bump' or 'all'.
    :return: Generator that returns the length of the specified category contained in each skeleton image.
    """
    for i, img in enumerate(image_lst):
        yield imptools.get_length2(genr_categ_skel(img, category))


def genr_categ_skel(img: ProcessedImage,
                    category: str,
                    height_thresh: tuple = (2.02, 3.78)) -> np.array:
    """
    :param img: Instance of ProcessedImage class
    :param category: Category for which you want to generate skeleton image.
                     Category should be set to either 'dent', 'kink','ep','straight','kinked_end', 'normal','bump' or 'all'.
    :param height_thresh: height threshold for dent and bump
    :return: skeleton image for category you chose.
    """
    if category in {'kink', 'ep', 'straight', 'kinked_end'}:
        # 各カテゴリーのラベルを含むdentの細線画像を返す。ラベルイメージが必要
        # 別の関数としてまとめた方が良いし、dentの形状のトラッキングのために、xtrackとかの解析が必要かも？
        # そういうのはfiberクラスに実装しても良いかもしれない。その場合,epのラベルをfiberインスタンスが含んでいるのか調べる必要がある。
        # なんにせよdentの各ラベルの形状をトラッキングできるようにしたい
        return genr_dent_categ_skel(img, category, height_thresh)

    elif category == 'dent':  # todo skeleton_imageはbpを除去しなくても良い？
        dent_skel = img.skeleton_image & (img.calibrated_image < height_thresh[0])  # 下限値0.3は不要？
        return dent_skel

    elif category == 'normal':
        normal_skel = img.skeleton_image & (height_thresh[0] <= img.calibrated_image) & (img.calibrated_image <= height_thresh[1])
        return normal_skel

    elif category == 'bump':
        bump_skel = img.skeleton_image & (height_thresh[1] <= img.calibrated_image)
        return bump_skel

    elif category == 'all':
        return img.skeleton_image


def genr_dent_categ_skel(img: ProcessedImage,
                         dent_category: str,
                         height_thresh: tuple = (2.02, 3.78)) -> np.array:
    """
    :param img:
    :param dent_category: Category of dents. This value should be set to one of 'kink','ep','straight','kinked_end'
    :param height_thresh:
    :return: skeleton image of dents which contains
    """
    # todo skeleton_imageはbpを除去しなくても良い？
    # todo size=1のやつは除去しなくて良い？
    dent_skel = (img.skeleton_image & (img.calibrated_image < height_thresh[0])).astype(np.uint8)  # todo 下限値0.3は不要？
    dent_nlabel, dent_labelimage = cv2.connectedComponents(dent_skel)
    kink_image = np.zeros((1024, 1024)).astype(np.uint8)  # todo 画像のサイズが変わったら動かない。要対応
    kink_image[img.kink_positions[0], img.kink_positions[1]] = 1
    ep_image = imptools.endPoints(img.skeleton_image)
    all_label = np.arange(1, dent_nlabel)
    all_kink_label = np.setdiff1d(np.unique(dent_labelimage * kink_image), np.array([0]))
    all_ep_label = np.setdiff1d(np.unique(dent_labelimage * ep_image), np.array([0]))
    kink_or_ep_label = np.union1d(all_kink_label, all_ep_label)
    category_label = {'kink'      : np.setdiff1d(all_kink_label, all_ep_label),
                      'ep'        : np.setdiff1d(all_ep_label, all_kink_label),
                      'straight'  : np.setdiff1d(all_label, kink_or_ep_label),
                      'kinked_end': np.intersect1d(all_kink_label, all_ep_label)}[dent_category]

    d_categ_skel = np.where(np.isin(dent_labelimage, category_label), 1, 0).astype(np.uint8)
    return d_categ_skel


def calc_dent_categ_dict(img: ProcessedImage,
                         height_thresh: tuple = (2.02, 3.78)) -> Tuple[np.array, Dict]:
    """
    :param img:
    :param dent_category: Category of dents. This value should be set to one of 'kink','ep','straight','kinked_end'
    :param height_thresh:
    :return: skeleton image of dents which contains
    """
    # todo skeleton_imageはbpを除去しなくても良い？
    # todo size=1のやつは除去しなくて良い？
    dent_skel = (img.skeleton_image & (img.calibrated_image < height_thresh[0])).astype(np.uint8)  # todo 下限値0.3は不要？
    dent_nlabel, dent_labelimage = cv2.connectedComponents(dent_skel)
    kink_image = np.zeros((1024, 1024)).astype(np.uint8)  # todo 画像のサイズが変わったら動かない。要対応
    kink_image[img.kink_positions[0], img.kink_positions[1]] = 1
    ep_image = imptools.endPoints(img.skeleton_image)
    all_label = np.arange(1, dent_nlabel)
    all_kink_label = np.setdiff1d(np.unique(dent_labelimage * kink_image), np.array([0]))
    all_ep_label = np.setdiff1d(np.unique(dent_labelimage * ep_image), np.array([0]))
    kink_or_ep_label = np.union1d(all_kink_label, all_ep_label)
    category_label_dict = {'kink'      : np.setdiff1d(all_kink_label, all_ep_label),
                           'ep'        : np.setdiff1d(all_ep_label, all_kink_label),
                           'straight'  : np.setdiff1d(all_label, kink_or_ep_label),
                           'kinked_end': np.intersect1d(all_kink_label, all_ep_label)}
    return dent_labelimage, category_label_dict


def save_dent_category_image(img : ProcessedImage,
                             dent_label_img : np.array,
                             categ_label_dict : dict,
                             savename : Path):

    category_colors = {'kink': 'coral',
                       'ep': 'olive',
                       'straight': 'cadetblue',
                       'kinked_end': 'seagreen'}
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img.binarized_image, cmap='gray')
    ax.axis('off')
    for category, labels in categ_label_dict.items():
        categ_posi = np.where(np.isin(dent_label_img, labels))
        ax.scatter(categ_posi[1], categ_posi[0],
                   s=10, c=category_colors[category])
    plt.savefig(savename, dpi=900)


project_dir = Path('/Users/tomok/Documents/Python/FirstPaperProgram')
data_name_dir = project_dir / 'pickle_data/'
figure_dir = project_dir / 'Figure/'

if __name__ == '__main__':
    data_name_dict = {n: file.stem for n, file in enumerate(data_name_dir.iterdir()) if file.is_dir()}
    pprint.pprint(data_name_dict)
    data_key = int(input('choose the data to analyse: '))
    print('this process take a long time. please wait.')

    # make directory for save images for dent analysis
    dent_categ_image_dir = figure_dir / (data_name_dict[data_key]) / 'dent_category_image'
    if not dent_categ_image_dir.exists():
        dent_categ_image_dir.mkdir()

    image_data_dir = data_name_dir / (data_name_dict[data_key])
    image_objects_path = image_data_dir.glob('*.pickle')
    images = []
    for file in image_objects_path:
        with open(file, 'rb') as f:
            image = pickle.load(f)
        images.append(image)
        dent_label_image, categoly_label_dict = calc_dent_categ_dict(image)
        #  todo 一つ上の行の変数を作らないと画像保存できない仕様が嫌い。要改善
        save_dent_category_image(image,
                                 dent_label_image,
                                 categoly_label_dict,
                                 dent_categ_image_dir / (file.stem + '.png'))

    breakdown_dict = calc_breakdown(images)
    print(breakdown_dict)
    # make directory for save breakdown of dent(each category), normal, bump
    breakdown_data_dir = data_name_dir / (data_name_dict[data_key]) / 'breakdown'
    if not breakdown_data_dir.exists():
        breakdown_data_dir.mkdir()
    with open(breakdown_data_dir / 'breakdown_dict_data.pickle', 'wb') as bd:
        pickle.dump(breakdown_dict, bd)
