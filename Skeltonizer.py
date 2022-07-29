from skimage.morphology import skeletonize, thin
import numpy as np
import imptools
import matplotlib.pyplot as plt
import cv2
import traceback


class Skeltonizer:
    def __init__(self, bp_height=5, branch_length=8, min_area=10, image_size=1024):
        self.bp_height = bp_height
        self.branch_length = branch_length
        self.min_area = min_area
        self.image_size = image_size
        # インスタンス変数のメリットは、複数回利用するような変数を保持できる事？だとしたら今のインスタンス変数はあまり意味がない
        # 内部処理用の変数として、coor_low_bps, coor_close_eps, branches_imageなどは保存していた方が、
        # 説明用のグラフ作ったりバグ修正に便利かも
        self._coor_low_bps = None
        self._coor_high_bps = None
        self._coor_close_eps = None
        self._branches_image = None

        self._init_skeleton_image = None

    def __call__(self, image):
        """
        -------------input-------------------
        image:
            BGCalibrater -> Segmentater で出力したProcessedImageインスタンス
        -------------output-------------------
        入力のProcessedImageインスタンスの属性に細線化画像（枝除去済み）を追加する

        -------------processing flow-----------------
        extract initial center line (with branch)
        remove branch:
            init_skeleton - branch_image
            pick up low branch point:

            prune short branch derive from low branch point:
        """

        init_skeleton_image = thin(image.binarized_image)
        self._init_skeleton_image = init_skeleton_image
        self.set_low_bp_coor(image.calibrated_image, init_skeleton_image, self.bp_height)
        self.get_close_eps()
        nobranch_skeleton_image = self.prune_branches(image.calibrated_image, init_skeleton_image)
        nobranch_skeleton_image = skeletonize(nobranch_skeleton_image).astype(np.uint8)
        nobranch_skeleton_image = self.remove_small_and_ring(nobranch_skeleton_image)
        nLabels, label_Images, data, center = cv2.connectedComponentsWithStats(nobranch_skeleton_image)
        image.skeleton_image = nobranch_skeleton_image
        image.label_image = label_Images
        image.nLabels = nLabels
        image.data = data

    def prune_branches(self,
                       calibrated_image,
                       init_skeleton_image):
        branches_image = self.calc_branches_image(calibrated_image, init_skeleton_image)
        return init_skeleton_image - branches_image

    def calc_branches_image(self,
                            calibrated_image,
                            init_skeleton_image):
        #         coor_low_bps = np.where(imptools.branchedPoints(init_skeleton_image))
        branches_image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        coor_branch = self.track_branches()
        branches_image[coor_branch] = 1
        return branches_image

    def set_low_bp_coor(self,
                        calibrated_image,
                        init_skeleton_image,
                        bp_height):
        all_bps = imptools.branchedPoints(init_skeleton_image)
        low_bp_coor = np.where(all_bps & (calibrated_image < bp_height))
        high_bp_coor = np.where(all_bps & (calibrated_image >= bp_height))
        self._coor_low_bps = low_bp_coor
        self._coor_high_bps = high_bp_coor

    def get_close_eps(self):  # eps eny close than branch_length
        close_eps = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        all_eps_coorx, all_eps_coory = np.where(imptools.endPoints(self._init_skeleton_image))
        _low_bps_image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        _low_bps_image[self._coor_low_bps] = 1
        for x, y in zip(all_eps_coorx, all_eps_coory):
            window = _low_bps_image[x - self.branch_length : x + self.branch_length,
                                    y - self.branch_length : y + self.branch_length]
            if window.any():  # window中に一つでもlow_bpが入っているならTrue
                close_eps[x, y] = 1

        coor_close_eps = np.where(close_eps)
        self._coor_close_eps = coor_close_eps

    def track_branches(self):
        branches_coor_x = []
        branches_coor_y = []
        image_for_tracking = self._init_skeleton_image.copy()

        image_low_bps = np.zeros((self.image_size, self.image_size), dtype=bool)
        image_low_bps[self._coor_low_bps] = True  # low_bpに目印として2を設定

        image_high_bps = np.zeros((self.image_size, self.image_size), dtype=bool)
        image_high_bps[self._coor_high_bps] = True
        # epからトラック開始。low_bpにぶつかったら終了
        starts_x, starts_y = self._coor_close_eps
        for step_num, (start_x, start_y) in enumerate(zip(starts_x, starts_y)):
            # tracking_area中の座標からimage_for_tracking中の座標に変換するため、x,yを最後に足すの忘れずに！

            tracking_area = image_for_tracking[start_x - self.branch_length : start_x + self.branch_length,
                                               start_y - self.branch_length : start_y + self.branch_length]

            image_for_low_bp_detection = image_low_bps[start_x - self.branch_length: start_x + self.branch_length,
                                                   start_y - self.branch_length: start_y + self.branch_length]

            image_for_high_bp_detection = image_high_bps[start_x - self.branch_length: start_x + self.branch_length,
                                                         start_y - self.branch_length: start_y + self.branch_length]

            x, y = self.branch_length, self.branch_length  # トラッキング開始点となるtracking_areaの中心点の位置

            xtrack = [x + start_x - self.branch_length]  # 各枝ごとの結果格納用(初期位置であるtracking_areaの中心は最初に追加)
            ytrack = [y + start_y - self.branch_length]

            try:
                for i in range(self.branch_length):
                    # bpに辿り着かなかった場合の終判定どうなってる？
                    tracking_area[x, y] = 0  # 現在の注目画素の値を0に更新
                    # 移動方向の探索
                    window = tracking_area[x - 1:x + 2, y - 1:y + 2]
                    if (window == 0).all():  # windowsの中身が全て0(つまり別のepにたどり着いた)なら停止.極端に短いCNFを想定している？
                        branches_coor_x += xtrack
                        branches_coor_y += ytrack
                        break
                    elif image_for_low_bp_detection[x - 1:x + 2, y - 1:y + 2].any():  # low_branchの手前になったら消去。
                        branches_coor_x += xtrack
                        branches_coor_y += ytrack
                        break
                    elif image_for_high_bp_detection[x - 1:x + 2, y - 1:y + 2].any():
                        # lowではないepにぶつかった場合の終了判定が必要
                        # tracked line is not counted as branch when high bp is detected
                        break
                    direction = np.where(window != 0)
                    direction = [a - 1 for a in direction]
                    x += int(direction[0])
                    y += int(direction[1])
                    xtrack.append(x + start_x - self.branch_length)
                    ytrack.append(y + start_y - self.branch_length)


            except TypeError:
                print('something wrong with branch removal in Skeletonizer')
                traceback.print_exc()

                fig, ax = plt.subplots(1, 5, figsize=(15, 3))
                for a in ax:
                    a.axis('off')
                ax[0].set_title('current state')
                ax[0].imshow(tracking_area, cmap='gray')
                ax[1].set_title('initial state')
                ax[1].imshow(self._init_skeleton_image[start_x - self.branch_length: start_x + self.branch_length,
                                                       start_y - self.branch_length: start_y + self.branch_length],
                              cmap='gray')
                ax[2].set_title('searching window')
                ax[2].imshow(window, cmap='gray')
                ax[3].set_title('low bp position')
                ax[3].imshow(image_for_low_bp_detection, cmap='gray')
                ax[4].set_title('high bp position')
                ax[4].imshow(image_for_high_bp_detection, cmap='gray')
                plt.show()

        branches_coor_x = np.asarray(branches_coor_x)
        branches_coor_y = np.asarray(branches_coor_y)
        return branches_coor_x, branches_coor_y

    def remove_small_and_ring(self, skeleton_image):
        nLabels, label_Images, data, center = cv2.connectedComponentsWithStats(skeleton_image)
        bp = imptools.endPoints(skeleton_image)
        ring_frac_label = np.setdiff1d(np.arange(1, nLabels), (bp * label_Images))
        for i in range(1, nLabels):
            area = data[i][4]
            if area < self.min_area:
                skeleton_image[np.where(label_Images == i)] = 0

        for i in ring_frac_label:
            skeleton_image[np.where(label_Images == i)] = 0
        return skeleton_image


