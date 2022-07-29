import imptools
import numpy as np
import cv2
import matplotlib.pyplot as plt


class KinkDetecter:
    def __init__(self,
                 N=5,
                 distance_angle_diff=10,
                 angle_threshold=30,
                 min_area=10):  # なんかいい感じの引数名あったらいいな

        self.N = N  # mean angle for [i: i+N] is calculated
        self.distance_calc_angle_diff = distance_angle_diff
        self.angle_threshold = angle_threshold
        self.min_area = min_area # fraction smaller than this value was ignored after removal of bp

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
        image_divided_bp = self.remove_small(image_divided_bp, min_area=10)
        nLabels, label_Images, data, center = cv2.connectedComponentsWithStats(image_divided_bp)

        kink_posis_x = []
        kink_posis_y = []
        for n in range(1, nLabels):  # note: index-0 is background
            # 最後にx,y座標にxとyを足すの忘れずに！
            # もうimage_size**2の画像作って判定した方が色々楽そうなのでそうする
            y, x, h, w, area = data[n]
            target = image_divided_bp[x: x + w, y: y + h]  # 注目中のCNF

            # test = target[0:7, -10:-1]
            # ep = imptools.test_endPoints(test)
            # print(ep)
            # # print(np.where(imptools.test_endPoints(target)))
            # # print(target[0:7, -10:-1])
            # # print(imptools.test_endPoints(target)[0:7, -10:-1])
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(target, cmap='gray')
            # ax[1].imshow(image.skeleton_image[x: x + w, y: y + h], cmap='gray')
            # ax[2].imshow(target + imptools.endPoints(target), cmap='gray')
            # plt.show()

            xtrack, ytrack = imptools.tracking2(target)
            print(xtrack)

            angle_list = self.calc_angle_list(target)
            mean_angle_list = self.calc_mean_angle_list(angle_list)
            angle_diffs = mean_angle_list - np.roll(mean_angle_list, self.distance_calc_angle_diff)
            kink_candidate = np.where(np.abs(angle_diffs) > self.angle_threshold)
            kink_posi_index = self.remove_close_kink(kink_candidate)

            kink_posis_x += xtrack[kink_posi_index]
            kink_posis_y += ytrack[kink_posi_index]
            transformed_x = self.trans_coor(kink_posis_x)
            transformed_y = self.trans_coor(kink_posis_y)

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
        #         angle_list = np.arctan2(np.roll(ytrack,1)-ytrack,
        #                                                   xtrack-np.roll(xtrack,1))    #引き算の順序逆では？ githubで修正前後のバージョン作ってみる
        angle_list = np.rad2deg(angle_list)
        return angle_list

    def calc_mean_angle_list(self, angle_list):
        mean_conv_filter = np.ones(self.N)/self.N
        mean_angle_list = np.convolve(angle_list, mean_conv_filter, mode='same')
        return mean_angle_list


    @staticmethod
    def remove_close_kink(kink_posi_cand):
        return kink_posi
