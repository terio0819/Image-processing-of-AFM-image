from original_modules import imptools
import numpy as np
import cv2
import matplotlib.pyplot as plt
import traceback


class KinkDetector:
    def __init__(self,
                 N=5,
                 distance_angle_diff=10,
                 angle_threshold=30,
                 min_area=20,
                 gap=10):  # なんかいい感じの引数名あったらいいな

        self.N = N  # mean angle for [i: i+N] is calculated
        self.distance_calc_angle_diff = distance_angle_diff
        self.angle_threshold = angle_threshold
        self.min_area = min_area # fraction smaller than this value was ignored after removal of bp
        self.gap = gap

    def __call__(self, image):
        """

        :param image:
        :return:
        """
        image.bp = imptools.branchedPoints(image.skeleton_image)
        image.ep = imptools.endPoints(image.skeleton_image)
        image.kink_positions = self.calc_kink_positions(image)

    def calc_kink_positions(self, image):
        """

        :param image:
        :return:
        """

        skeleton_image = image.skeleton_image

        image_divided_bp = self.remove_bp(skeleton_image)
        image_divided_bp = self.remove_small(image_divided_bp, min_area=self.min_area)
        nLabels, label_Images, data, center = cv2.connectedComponentsWithStats(image_divided_bp)

        _kink_cand_x = []
        _kink_cand_y = []
        kink_posis_x = []
        kink_posis_y = []
        for i in range(1, nLabels):  # note: index-0 is background

            y, x, h, w, area = data[i]
            target = np.zeros_like(image_divided_bp)
            target[np.where(label_Images == i)] = 1
            try:
                xtrack, ytrack = imptools.tracking2(target)
                angle_list = self.calc_angle_list(target)
                mean_angle_list = self.calc_mean_angle_list(angle_list)
                angle_diffs = np.abs(mean_angle_list - np.roll(mean_angle_list, self.distance_calc_angle_diff))
                kink_candidate, *_ = np.where(angle_diffs > self.angle_threshold)
                kink_candidate = kink_candidate[kink_candidate > self.distance_calc_angle_diff]
                # note: angle_diffs[0:self.distance_calc_angle_diff] is meaningless
                kink_posi_index = self.integrate_close_kinks(angle_diffs, kink_candidate)

                _kink_cand_x += list(xtrack[kink_candidate])
                _kink_cand_y += list(ytrack[kink_candidate])

                if kink_posi_index:
                    kink_posis_x += list(xtrack[kink_posi_index])
                    kink_posis_y += list(ytrack[kink_posi_index])
            except:
                print('something wrong with kink detection.\nTo resume, close a figure.')
                traceback.print_exc()
                print(kink_candidate)
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(target[x: x + w, y: y + h], cmap='gray')
                ax[1].imshow(image.skeleton_image[x: x + w, y: y + h], cmap='gray')
                ax[2].imshow((target + imptools.endPoints(target))[x: x + w, y: y + h], cmap='gray')
                plt.show()

        image._kink_candidate = (np.array(_kink_cand_x), np.array(_kink_cand_y))
        kink_positions = (np.array(kink_posis_x), np.array(kink_posis_y))
        return kink_positions

    @staticmethod
    def remove_bp(skeleton_image, remove_size=1):
        imgcopy = skeleton_image.copy()
        bp = imptools.branchedPoints(imgcopy)
        bp_coor = np.where(bp)
        for bp_x, bp_y in zip(bp_coor[0], bp_coor[1]):  # bpの周囲を除去
            imgcopy[bp_x - remove_size:bp_x + remove_size + 1, bp_y - remove_size:bp_y + remove_size + 1] = 0
        return imgcopy

    @staticmethod
    def remove_small(skeleton_image, min_area):
        nLabels, label_Images, data, center = cv2.connectedComponentsWithStats(skeleton_image)
        for i in range(1, nLabels):
            area = data[i][4]
            if area < min_area:
                skeleton_image[np.where(label_Images == i)] = 0
        return skeleton_image

    @staticmethod
    def calc_angle_list(clipped_img):
        xtrack, ytrack = imptools.tracking2(clipped_img)
        angle_list = np.arctan2(np.roll(ytrack, 1) - ytrack,
                                np.roll(xtrack, 1) - xtrack)
        #  angle_list = np.arctan2(np.roll(ytrack,1)-ytrack,
        #                          xtrack-np.roll(xtrack,1))    #引き算の順序逆では？ githubで修正前後のバージョン作ってみる
        angle_list = np.rad2deg(angle_list)
        return angle_list

    def calc_mean_angle_list(self, angle_list):
        """
        reference https://zenn.dev/bluepost/articles/1b7b580ab54e95
        :param angle_list:
        :return:
        """
        mean_conv_filter = np.ones(self.N)/self.N
        mean_angle_list = np.convolve(angle_list, mean_conv_filter, mode='same')

        n_conv = self.N // 2
        mean_angle_list[0] *= self.N/n_conv
        for i in range(1, n_conv):
            mean_angle_list[i] *= self.N/(i + n_conv)
            mean_angle_list[-i] *= self.N/(i + n_conv - (self.N % 2))

        return mean_angle_list


    def integrate_close_kinks(self, angle_diffs, kink_candidate):
        """

        :param angle_diffs:
        :param kink_candidate:
        :return:
        """
        if len(kink_candidate) == 0:
            return []
        if len(kink_candidate) == 1:
            return list(kink_candidate)

        diff_between_indices = (np.roll(kink_candidate, -1) - kink_candidate)[:-1]
        close_kinks_starts = []  # 近いピーク集団のスタート位置を表す(diff_between_indicesの中での)インデックスの集合
        close_kinks_ends = []

        integrated_kink_candidates = []
        if diff_between_indices[0] < self.gap:
            close_kinks_starts.append(0)

        for i in range(1, len(diff_between_indices)):
            if diff_between_indices[i - 1] > self.gap and diff_between_indices[i] > self.gap:  # 独立したピークを返値に入れる
                integrated_kink_candidates.append(kink_candidate[i])

            if diff_between_indices[i - 1] > self.gap and diff_between_indices[i] <= self.gap:
                close_kinks_starts.append(i)

            if diff_between_indices[i - 1] <= self.gap and diff_between_indices[i] > self.gap:
                close_kinks_ends.append(i)

        if diff_between_indices[-1] <= self.gap:
            close_kinks_ends.append(len(diff_between_indices)-1)  #本当は-1不要かも

        start_end_pairs = [(s, e) for s, e in zip(close_kinks_starts, close_kinks_ends)]

        # integrate close kink to the sharpest
        for (s, e) in start_end_pairs:
            sharpest_kink_index = max(range(s, e + 1), key=lambda x: angle_diffs[kink_candidate[x]])
            integrated_kink_candidates.append(kink_candidate[sharpest_kink_index])

        return sorted(integrated_kink_candidates)
