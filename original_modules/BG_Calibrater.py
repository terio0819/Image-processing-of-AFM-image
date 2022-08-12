import cv2
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class BG_Calibrater:

    def __init__(self, savgol_window=31, savgol_polyorder=2, apply_median=True):
        """

        :param savgol_window: int
        :param savgol_polyorder: int
        :param apply_median: boolean
        """

        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.apply_median = apply_median

    def __call__(self, image):
        """

        :param image: ProcessedImage class instance

        :return: Image class instance with attribution of calibrated_image
        """

        thresholds_low, thresholds_high = self._get_thresholds(image.original_image)
        # 以下のfilterは、ただ中間結果を保持しているだけなので、改善できる気がする
        background_filter = self._get_background_filter(image.original_image, thresholds_low, thresholds_high)
        background_splined = self._get_splined_background(image.original_image, background_filter)
        background_smoothed = self._savgol_smoothing(background_splined, self.savgol_window, self.savgol_polyorder)
        calibrated_image = image.original_image - background_smoothed
        if self.apply_median:
            calibrated_image = cv2.medianBlur(calibrated_image.astype(np.float32), ksize=3)
            
        image.calibrated_image = calibrated_image

    @staticmethod
    def _get_thresholds(image):      # 各行の外れ値の閾値を取得（2次元array）
        """

        :param image: ndarray shape=(ImageSize, ImageSize)

        :return:
        """
        weight_q1 = stats.scoreatpercentile(image, 25, axis=1).reshape((image.shape[0], 1))
        weight_q3 = stats.scoreatpercentile(image, 75, axis=1).reshape((image.shape[0], 1))
        weight_iqr = weight_q3 - weight_q1
        thresholds_low = weight_q1 - weight_iqr * 1.5
        thresholds_high = weight_q3 + weight_iqr * 0
        return thresholds_low, thresholds_high
    
    @staticmethod
    def _get_background_filter(image, thresholds_low, thresholds_high):

        """
        get the filter to remove the outlier pixel(CNF and depression) from AFM image

        :param image: ndim shape = (ImageSize, ImageSize)
        :param thresholds_low: ndim shape = (ImageSize, 1)
        :param thresholds_high: ndim shape = (ImageSize, 1)

        :return:
        """
        background_filter = (image > thresholds_low) & (image < thresholds_high)
        background_filter[:, 0] = True
        background_filter[:, -1] = True
        return background_filter

    @staticmethod
    def _get_splined_background(original_image,
                                background_filter):

        """
        interpolate the splined background after removing outlier

        :param original_image:
        :param background_filter:
        :return:
        """
        splined_bg = np.empty(original_image.shape)
        n_row, n_column = original_image.shape
        for row in range(n_row):
            filter_row = background_filter[row]
            background_x, *_ = np.where(filter_row)
            background_height = original_image[row][filter_row]
            f = interp1d(background_x, background_height)
            x = np.arange(n_column)  # 画像の横一列のピクセル番号（0～1023）
            splined_bg[row] = f(x)
        return splined_bg

    @staticmethod
    def _savgol_smoothing(background_splined, window_length, polyorder):
        def savgol(row, wl=window_length, po=polyorder):
            return savgol_filter(row, wl, po)
        bg_smoothed = np.apply_along_axis(savgol, 1, background_splined)
        return bg_smoothed
