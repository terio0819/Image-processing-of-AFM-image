import cv2
from Image_class import ProcessedImage, Fiber
import imptools
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Dict, TypedDict, Generator, Tuple


class DentDict(TypedDict):
    kink: float
    ep: float
    straight: float
    kinked_end: float


class BreakdownDict(TypedDict):
    dent: DentDict
    normal: float
    bump: float


class Dent_DepthLength_Dict(TypedDict):
    kink: Tuple[np.ndarray]
    ep: Tuple[np.ndarray]
    straight: Tuple[np.ndarray]
    kinked_end: Tuple[np.ndarray]


class DentAnalyzer:
    # todo call関数でimageを処理して、dentのlabelimageとlabelの情報をProcessed_Imageクラスの属性として保存しておく仕様にするべきだった。
    # 時間が無いので諦める。
    def __init__(self,  # todo 現状イニシャライザに意味はない。将来的にここから値を参照する様に改善する
                 upper_thresh=3.78,
                 lower_thresh=2.02):
        self.upper_thresh = upper_thresh
        self.lower_thresh = lower_thresh

    def __call__(self, image: ProcessedImage):
        dent_skel_labelimg, categorized_dent_labeldict = self._calc_dent_categ_dict(image)
        image.dent_skel_labelimg = dent_skel_labelimg
        image.categorized_dent_labeldict = categorized_dent_labeldict

    def calc_breakdown(self,
                       image_list: List[ProcessedImage]) -> BreakdownDict:
        out_dict = BreakdownDict(dent=DentDict())
        categories = {'dent', 'normal', 'bump'}
        dent_categories = {'kink', 'ep', 'straight', 'kinked_end'}
        all_length = sum(self._calc_categ_length(image_list, 'all'))
        for cat in categories:
            if cat == 'dent':
                for d_cat in dent_categories:
                    d_cat_component = sum(self._calc_categ_length(image_list, d_cat)) / all_length
                    out_dict['dent'][d_cat] = d_cat_component
            else:
                cat_component = sum(self._calc_categ_length(image_list, cat)) / all_length
                out_dict[cat] = cat_component

        # todo out_dictの値の合計が、微妙に1にならない問題がある。多分丸め込み誤差？
        #  正規化して,さらに合計100になるようにしておく。
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

    def calc_dl(self,
                image_list: List[ProcessedImage]) -> Dent_DepthLength_Dict:
        dl_dict = Dent_DepthLength_Dict()  # todo インスタンス生成の段階で、関数を呼び出す様にすれば良いのでは

        dent_categories = ('kink', 'ep', 'straight', 'kinked_end')

        for category in dent_categories:
            dl_dict[category] = (list(self._calc_dent_depth(image_list, category)),
                                 list(self._calc_dent_length(image_list, category)))
        return dl_dict

    @staticmethod
    def _calc_dent_depth(image_list: List[ProcessedImage],
                         category: str) -> Generator:
        for img in image_list:
            for label in (img.categorized_dent_labeldict[category]):
                depth = 2.02 - np.min(img.calibrated_image[np.where(img.dent_skel_labelimg == label)])
                yield depth

    @staticmethod
    def _calc_dent_length(image_list: List[ProcessedImage],
                          category: str) -> Generator:
        for img in image_list:
            for label in (img.categorized_dent_labeldict[category]):
                length = imptools.get_length2(np.where(img.dent_skel_labelimg == label, 1, 0))
                yield length

    @staticmethod
    def save_dent_category_image(img: ProcessedImage,
                                 dent_label_img: np.array,
                                 categ_label_dict: dict,
                                 savename: Path):

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

    def _calc_categ_length(self,
                           image_lst: List[ProcessedImage],
                           category: str) -> Generator[float, None, None]:
        """
        :param image_lst:
        :param category: Category for which you want to generate skeleton image.
                         Category should be set to either 'dent', 'kink','ep','straight','kinked_end', 'normal','bump' or 'all'.
        :return: Generator that returns the length of the specified category contained in each skeleton image.
        """
        for i, img in enumerate(image_lst):
            yield imptools.get_length2(self._genr_categ_skel(img, category))

    def _genr_categ_skel(self,
                         img: ProcessedImage,
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
            return self._genr_dent_categ_skel(img, category)

        elif category == 'dent':  # todo skeleton_imageはbpを除去しなくても良い？
            dent_skel = img.skeleton_image & (img.calibrated_image < height_thresh[0])  # 下限値0.3は不要？
            return dent_skel

        elif category == 'normal':
            normal_skel = img.skeleton_image & (height_thresh[0] <= img.calibrated_image) & (
                    img.calibrated_image <= height_thresh[1])
            return normal_skel

        elif category == 'bump':
            bump_skel = img.skeleton_image & (height_thresh[1] <= img.calibrated_image)
            return bump_skel

        elif category == 'all':
            return img.skeleton_image

    def _genr_dent_categ_skel(self,
                              img: ProcessedImage,
                              dent_category: str) -> np.array:
        """
        :param img:
        :param dent_category: Category of dents. This value should be set to one of 'kink','ep','straight','kinked_end'
        :return: skeleton image of dents which contains
        """
        # todo skeleton_imageはbpを除去しなくても良い？
        # todo size=1のやつは除去しなくて良い？
        category_label = img.categorized_dent_labeldict[dent_category]
        d_categ_skel = np.where(np.isin(img.dent_skel_labelimg, category_label), 1, 0).astype(np.uint8)
        return d_categ_skel

    @staticmethod
    def _calc_dent_categ_dict(img: ProcessedImage,
                              height_thresh: tuple = (2.02, 3.78)) -> Tuple[np.array, Dict]:
        """
        :param img:
        :param dent_category: Category of dents. This value should be set to one of 'kink','ep','straight','kinked_end'
        :param height_thresh:
        :return: skeleton image of dents which contains
        """
        # todo skeleton_imageはbpを除去しなくても良い？
        # todo size=1のやつは除去しなくて良い？
        dent_skel = (img.skeleton_image & (img.calibrated_image < height_thresh[0])).astype(np.uint8)
        dent_nlabel, dent_labelimage = cv2.connectedComponents(dent_skel)
        kink_image = np.zeros((1024, 1024)).astype(np.uint8)  # todo 画像のサイズが変わったら動かない。要対応
        kink_image[img.kink_positions[0], img.kink_positions[1]] = 1
        ep_image = imptools.endPoints(img.skeleton_image)
        all_label = np.arange(1, dent_nlabel)
        all_kink_label = np.setdiff1d(np.unique(dent_labelimage * kink_image), np.array([0]))
        all_ep_label = np.setdiff1d(np.unique(dent_labelimage * ep_image), np.array([0]))
        kink_or_ep_label = np.union1d(all_kink_label, all_ep_label)
        category_label_dict = {'kink': np.setdiff1d(all_kink_label, all_ep_label),
                               'ep': np.setdiff1d(all_ep_label, all_kink_label),
                               'straight': np.setdiff1d(all_label, kink_or_ep_label),
                               'kinked_end': np.intersect1d(all_kink_label, all_ep_label)}
        return dent_labelimage, category_label_dict
