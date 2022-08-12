import cv2
import numpy as np
from skimage.feature import canny
from skimage.filters import threshold_local
from skimage.morphology import binary_opening, binary_closing, binary_erosion, binary_dilation
from skimage.transform import probabilistic_hough_line


class Segmentater:  # クッソ遅い（どこが遅いかは調べてないが、多分ラベリングとハフ変換）。本当にハフ変換必要なのか...
    def __init__(self,
                 area_min=200,
                 low_threshold=1.5,
                 global_threshold=0.3,
                 wsize_localbin=17,
                 h_length=12,
                 h_sratio=0.3):

        self.global_threshold = global_threshold
        self.wsize_localbin = wsize_localbin
        self.area_min = area_min
        self.low_threshold = low_threshold
        self.h_length = h_length
        self.h_sratio = h_sratio

        self.n_label = None
        self.no_linear = None

    def __call__(self, image):
        '''
        -------------input-------------------
        calibrated_image:
            BG_calibrater で出力したProcessedImageクラス
        -------------output-------------------

        -------------processing flow-----------------
        1. binary_global
        2. binary_local
        3. 1 & 2     ここまでひとまとめにで良さそう（binaryzation）
        4. remove small component     超ぐちゃぐちゃだけど、もっとシンプルにならないか？後で最大高さ1.5でふるいにかけるし
            4.1 smoothing by median filter  ぶっちゃけこれ必要か？とりあえず無視
            4.2 labeling
            4.3 remove small area under area_min これ必要か？4.5のやつをここでやれば良い気もする
            4.4 hough transformation　計算量多いので、省略できないか？
            4.5 erosion and remove small components under area_min2
            4.6 dilation
            4.7 closing

        '''

        binary_image = self._binaryzation(image.calibrated_image,
                                          self.global_threshold,
                                          self.wsize_localbin)

        no_small_binary_image = self._remove_connecting_fragments(binary_image,
                                                                  self.area_min)

        no_small_binary_image = self._remove_nonlinear_objects(no_small_binary_image,
                                                               self.h_length,
                                                               self.h_sratio)
        no_small_binary_image = self.remove_low_component(image.calibrated_image, no_small_binary_image)

        no_small_binary_image = binary_closing(no_small_binary_image)
        image.binarized_image = no_small_binary_image

    @staticmethod
    def _binaryzation(image, global_threshold, wsize_localbin):
        binary_global = image > global_threshold
        local_threshold = threshold_local(image, wsize_localbin)
        binary_local = image > local_threshold
        binary_final = binary_global & binary_local  # データ型はbooleanのままか、int8に直すか未定
        return binary_final

    @staticmethod
    # so heavy process
    def _remove_small(binary_image, area_min):
        out_binary_image = binary_image.copy()
        n_labels, label_image, stats, centers = cv2.connectedComponentsWithStats(np.uint8(binary_image), 8)
        for i in range(n_labels - 1):
            *_, area = stats[i]
            if area <= area_min:
                out_binary_image[label_image == i] = 0  # 入力のbinary_image自体を書き換えて出力
        return out_binary_image

    @staticmethod
    # オブジェクトが多すぎて、ラベルが0～255に収まりきらないかもしれん
    def _remove_connecting_fragments(binary_image, area_min):
        out_binary_image = binary_image.copy()
        out_binary_image = binary_erosion(out_binary_image)
        n_labels, label_image, stats, centers = cv2.connectedComponentsWithStats(np.uint8(binary_image), 8)
        for i in range(n_labels - 1):
            *_, area = stats[i]
            if area <= area_min:
                out_binary_image[label_image == i] = 0  # 入力のbinary_image自体を書き換えて出力
        out_binary_image = binary_dilation(out_binary_image)
        return out_binary_image

    @staticmethod
    def _remove_nonlinear_objects(binary_image, h_length, h_sratio, linegap=1):
        out_binary_image = binary_image.copy()
        n_labels, label_image, stats, centers = cv2.connectedComponentsWithStats(np.uint8(out_binary_image), 8)
        for i in range(n_labels - 1):
            left, top, width, height, area = stats[i]
            target = out_binary_image[top:top + height, left:left + width]  # インデックスの指定あってるか自信ない
            target_edge = canny(target, sigma=0,
                                low_threshold=0,
                                high_threshold=1)
            hough_lines = probabilistic_hough_line(target_edge,
                                                   line_length=h_length,
                                                   line_gap=linegap)
            line_lengths = [np.linalg.norm(np.array(start) - np.array(end)) for (start, end) in
                            hough_lines]  # あってるか自信ない
            total_length = sum(line_lengths)
            s_ratio = total_length / np.sum(target_edge)

            if s_ratio < h_sratio and np.sum(target) < 500:  # keep big object larger than 500
                out_binary_image[label_image == i] = 0

            return out_binary_image


    def remove_low_component(self, height_image, binary_image):
        n_labels, label_image, data, centers = cv2.connectedComponentsWithStats(np.uint8(binary_image), 8)
        for i in range(1, n_labels):
            max_height = np.max(height_image[np.where(label_image == i)])
            if max_height < self.low_threshold:
                binary_image[np.where(label_image == i)] = 0
        return binary_image
