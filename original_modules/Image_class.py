#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from original_modules import imptools
from pathlib import Path
import pickle
import pprint
import traceback
from typing import Iterable, List, Union
import math
import matplotlib.patches as patches
import warnings


class ProcessedImage:
    def __init__(self, original_AFM, name):
        """

        :param original_AFM: ndarray (ndim = 2)
        :param name:  str
        """
        self.name = name
        self.original_image = original_AFM
        self.binarized_image = None
        self.calibrated_image = None
        self.skeleton_image = None

        self.nLabels = None
        self.data = None  # y, x, h, w, area = data[label]
        self.label_image = None

        self.fiber_positions = None  # todo not implemented
        self.kink_positions: np.array = None
        self._kink_candidate = None  # todo not implemented

        self.bp = None
        self.ep = None

    def heights_skeleton(self):
        """
        画像中の細線部の高さ全部拾ってくるメソッド
        :return:
        """

        skeleton_posi = self.skeleton_image.astype(bool)
        heights_of_skeleton_lines = self.calibrated_image[skeleton_posi]
        return heights_of_skeleton_lines

    def all_length_distribution(self):
        """
        obtaine length distribution including isolated and overlapped CNF
        raise error if self.label_image == None
        """
        n_label = np.max(self.label_image)
        for label in range(n_label):
            all_length_distributions = imptools.get_length
        return all_length_distributions

    def isolated_length_distribution(self):
        """
        obtaine length distribution from isolated CNF
        """
        return isolated_length_distribution

    def average_length_distribution(self):
        return average_length_distribution

    def save_fiber_instances(self,
                             save_dir: pathlib.Path):
        # todo AFM resolution is not taken into account and needs to be improved in the future
        """
        Fiber class objects were saved in savedir/filename.stem
        :param save_dir: str
        :return:
        """

        fiber_dir = Path(f'{save_dir}/Fiber_data/{self.name}')
        if not fiber_dir.exists():
            fiber_dir.mkdir(parents=True)

        fiber_posi_map = self.calibrated_image
        fig, ax = plt.subplots(1, 1)
        ax.imshow(fiber_posi_map, cmap='afmhot', vmin=-0.5, vmax=5.5)
        ax.axis('off')

        nobp_image = imptools.remove_bp(self.skeleton_image)
        nLabels, label_Images, data, center = cv2.connectedComponentsWithStats(nobp_image)

        # generate kink position for ProcessedImage from Fiber instances
        # todo　別のメソッド、またはモジュールとしてこの作業は分離するようなリファクタリングが必要
        _kink_positions = [[],
                           []]
        for label in range(1, nLabels):
            x, y, w, h, size = data[label]
            r = patches.Rectangle(xy=(x, y), width=w, height=h, ec='white', fill=False, alpha=0.3)
            ax.add_patch(r)
            ax.annotate(label, xy=(x, y), color='white')

            try:
                target_image = np.where(label_Images == label, 1, 0)

                _xtrack, _ytrack = imptools.tracking2(target_image)
                xtrack = _xtrack - y
                # todo This is numerically correct, but super weird. I'll fix it when I can afford it.
                ytrack = _ytrack - x

                # calculate horizontal position array of each fiber
                xmove = xtrack - np.roll(xtrack, 1)
                ymove = ytrack - np.roll(ytrack, 1)
                xmove = np.delete(xmove, 0)
                ymove = np.delete(ymove, 0)  # index0の値は意味がないので消去
                a = xmove != 0
                b = ymove != 0  # x,y座標の変化の有無を表す配列(True:変化した　False:変化しなかった)
                c = np.vstack((a, b))
                d = np.all(c, axis=0)  # 進行方向を表す配列(True:斜め False:上下左右)

                horizon = [0]  # 0はスタート地点のx座標
                distance = 0
                for j in d:
                    if j == True:
                        distance += 2 * math.sqrt(2)
                    else:
                        distance += 2
                    horizon.append(distance)
                horizon = np.asarray(horizon)

                # calculate height array of each fiber
                height = self.calibrated_image[_xtrack, _ytrack]

                fiber_image = self.calibrated_image[y:y + h, x:x + w]

                # make Fiber instances
                fiber = Fiber(fiber_image, data[label], xtrack, ytrack, horizon, height)

                # detect and save kink position on each fiber
                coordinate = np.vstack((fiber.xtrack, fiber.ytrack))
                edge_indices = [0, len(fiber.xtrack) - 1] #  coordinateの最初と最後の位置ベクトルのインデックス
                fiber.set_binary_decomposition_indices(coordinate, edge_indices, dist_threshold=4)
                fiber.set_corner_indices(k=5, dist_thresh=5)
                fiber.set_kink_indices()

                # generate kink position for ProcessedImage from Fiber instances
                # todo　別のメソッド、またはモジュールとしてこの作業は分離するようなリファクタリングが必要
                _kink_positions[0].extend(_xtrack[fiber.kink_indices])
                _kink_positions[1].extend(_ytrack[fiber.kink_indices])

                with open(fiber_dir / f'{label}.pickle', mode='wb') as result:
                    pickle.dump(fiber, result)

            except:
                print('something wrong in save_fiber_instance')
                traceback.print_exc()

        # generate kink position for ProcessedImage from Fiber instances.
        # todo　別のメソッド、またはモジュールとしてこの作業は分離する
        warnings.simplefilter('error')
        try:
            self.kink_positions = np.array(_kink_positions)
        except:
            print('something is wrong with _kink_positions', _kink_positions)


        plt.savefig(fiber_dir / 'fiber_label.png')
        plt.close()


class Fiber:
    def __init__(self, fiber_image, data, xtrack, ytrack, hori_array, height_array):

        self.AFM_image = fiber_image
        self.data = data  # x, y, w, h, size of fiber in image.calibrated_imagee
        self.xtrack: np.array = xtrack  # x coordinates of fiber in self.AFM_image
        self.ytrack: np.array = ytrack
        self.horizon: np.array = hori_array
        self.height: np.array = height_array
        self.kink_hori_indices = None  # todo ProcessedImage.save_fiber_instanceで代入する予定だったが、まだやってない
        self.decomposition_indices: np.array = None  # kinkの候補点として使えそう
        self.corner_indices: np.array = None  # 検出制度がいまいち。使わないかも＿
        self.kink_indices: np.array = None

    def set_binary_decomposition_indices(self,
                                         line_coordinate: np.ndarray,
                                         edge_indices: List[int],
                                         dist_threshold: Union[int, float]) -> np.array:
        """

        :param line_coordinate:
        :param edge_indices:
        :param dist_threshold:
        :return:
        """
        # todo 最終的に得られる、edge_indicesの最初と最後は端点を表しているので、除去するべき
        new_edge_indices = self._add_farthest_indices(line_coordinate, edge_indices, dist_threshold)
        if new_edge_indices == edge_indices:
            # return new_edge_indices
            self.decomposition_indices = np.array(new_edge_indices)
        else:
            self.set_binary_decomposition_indices(line_coordinate, new_edge_indices, dist_threshold)

    def _add_farthest_indices(self,
                              line_coordinate: np.array,
                              edge_indices: List[int],
                              dist_threshold: Union[float, int]) -> List[int]:
        """

        :param line_coordinate:
        :param edge_indices:
        :param dist_threshold:
        :return:
        """
        new_edges = edge_indices.copy()
        for edge1, edge2 in zip(edge_indices, edge_indices[1:]):
            dist = self._calc_dists_from_line(edge1, edge2, line_coordinate)
            far_p_cand = np.argmax(dist)
            if dist[far_p_cand] >= dist_threshold:
                new_edges.append(far_p_cand + edge1)
        return sorted(new_edges)

    @staticmethod
    def _calc_dists_from_line(edge1_idx,
                              edge2_idx,
                              line_coor):  # todo 解像度が考慮されていないので、最後の出力で調整する？　dist_threshとか調整必要になるけど
        """
        calculate the distance between line edge1-edge2 and every point on curve.
        make sure that edge1 and edge2 is also on skel.
        :param edge1_idx: int 0 =< edge1 < edge2 < len(line_coor[0])
        :param edge2_idx: int 0 =< edge1 < edge2 < len(line_coor[0])
        :param line_coor: ndarray shape=(2, -1). This represents every coordinate of curve
        :return:
        """
        ab = line_coor[:, edge2_idx].reshape(-1, 1) - line_coor[:, edge1_idx].reshape(-1, 1)
        V = line_coor[:, edge1_idx:edge2_idx] - line_coor[:, edge1_idx].reshape(-1, 1)
        U = V - ab * ((ab.T @ V) / (ab.T @ ab))
        return np.linalg.norm(U, axis=0)

    def set_corner_indices(self, k, dist_thresh):
        """

        :param k:
        :param dist_thresh:
        :return:
        """
        coordinate = np.vstack((self.xtrack, self.ytrack))
        corner_indices = []
        for i in range(len(self.xtrack) - k):
            dist = self._calc_distance(i - k, i, i + k, coordinate)
            if dist >= dist_thresh:
                corner_indices.append(i)
        self.corner_indices = corner_indices

    @staticmethod
    def _calc_distance(idx0: int,
                       idx1: int,
                       idx2: int,
                       skel_coor: np.ndarray) -> float:
        """
        Calculate the distance between the line AC and point C.
        A, B, C are points on skeleton line represented by skel_coor
        :param idx0: skel_coor[:, idx0] corresponds to position vector of point A/
        :param idx1: int
        :param idx2: int. Those must be between 0 and the number of point consists skeleton_image.
        :param skel_coor: 2D-array (shape=(2,-1)). x and y coordinate of skeleton_image.
                          [[x_coordinate],
                           [y_coordinate]]
        :return:
        """
        ac = skel_coor[:, idx2].reshape(-1, 1) - skel_coor[:, idx0].reshape(-1, 1)
        ab = skel_coor[:, idx1].reshape(-1, 1) - skel_coor[:, idx0].reshape(-1, 1)
        ch = ac - ab * ((ab.T @ ac) / (ab.T @ ab))
        dist = np.linalg.norm(ch)
        return dist

    def set_kink_indices(self,    # todo 必ずset_binary_decompositionの後に実行しなければいけない。改善できる。
                         angle_thresh: Union[int, float] = 30) -> np.array:
        angle_arr = self._calc_angles(self.xtrack, self.ytrack, self.decomposition_indices)
        kink_indices = self.decomposition_indices[angle_arr >= angle_thresh]
        self.kink_indices = kink_indices

    @staticmethod
    def _calc_angles(xtrack: np.array,
                     ytrack: np.array,
                     decomposition_indices: Iterable[int]) -> np.array:
        """
        calculate angles between line AB and BC.
        A, B, C is adjacent decomposition points of skeleton line.
        (A = decomposition points[i], B = decomposition points[i+1], C = decomposition points[i+2])
        :param xtrack:　x-coordinate of skeleton line
        :param ytrack: y-coordinate of skeleton line
        :param decomposition_indices: indices of decomposition points on skeleton image
        :return: angle values of
        """
        if len(decomposition_indices) == 2:
            return np.array([0, 0])

        else:
            skel_coor = np.vstack((xtrack, ytrack))
            angle_arr = [0]  # this 0 is angle for first decomposition index.
            for i, j, k in zip(decomposition_indices, decomposition_indices[1:], decomposition_indices[2:]):
                # これi, j, kの３文字ではなく、i, i+1, i+2とした方が隣り合うdecomposition pointsの成す角ってことが伝わりやすかったかも
                ab = skel_coor[:, j].reshape(-1, 1) - skel_coor[:, i].reshape(-1, 1)
                bc = skel_coor[:, k].reshape(-1, 1) - skel_coor[:, j].reshape(-1, 1)
                cos_theta = (ab.T @ bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
                theta = np.arccos(cos_theta)[0, 0]
                theta = np.rad2deg(theta)
                angle_arr.append(theta)
            angle_arr.append(0)  # this 0 is angle for last decomposition index.
            return np.array(angle_arr)
