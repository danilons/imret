# coding: utf-8
import numpy as np
import cv2
from .relations import Relation

class RCC(object):
    rx = 64
    ry = 128
    ra = 192

    @property
    def x_value(self):
        return self.rx

    @property
    def y_value(self):
        return self.ry

    def compute(self, shape, object1, object2, kernel=None):

        imx = np.zeros(shape, dtype=np.float32)
        cv2.fillPoly(imx, [object1], 1)

        imy = np.zeros(shape, dtype=np.float32)
        cv2.fillPoly(imy, [object2], 2)

        return self.detect(imx, imy, kernel)

    def detect(self, imx, imy, kernel=None):
        hx, hy, ha = self._intersection(imx, imy)
        return self.decision(imx, imy, hx, hy, ha, kernel=kernel)

    def decision(self, imx, imy, hx, hy, ha, kernel=None):
        # define relations
        if hx == 0 and hy == 0 and ha != 0:
            return Relation(scope='EQ')

        if hx != 0 and hy != 0 and ha != 0:
            return Relation(scope='PO')

        # structuring element
        kernel = kernel or np.ones((11, 11), dtype=np.uint8)

        if hx == 0 and hy != 0 and ha != 0:
            # PP, but which one?
            imd = cv2.dilate(imx, kernel)
            dilation, _, _ = self._intersection(imd, imy)
            if dilation != 0:
                return Relation(scope='TPP')
            else:
                if not self.is_border(imx):
                    return Relation(scope='NTPP')
                return Relation(scope='TPP')

        if hx != 0 and hy == 0 and ha != 0:
            # PPi, but which one?
            imd = cv2.dilate(imy, kernel)
            _, dilation, _ = self._intersection(imx, imd)
            if dilation != 0:
                return Relation(scope='TPPi')
            else:
                if not self.is_border(imy):
                    return Relation(scope='NTPPi')
                return Relation(scope='TPPi')

        if hx != 0 and hy != 0 and ha == 0:
            imd = cv2.dilate(imx, kernel)
            _, _, dilation = self._intersection(imd, imy)
            if dilation != 0:
                return Relation(scope='EC')
            else:
                Relation(scope='DC')

        return Relation(scope='DC')

    def _intersection(self, ix, iy):
        """
        Compute region intersection from two binary images
        :param ix: image x
        :param iy: image y
        :return: area of regions (x, y, intersection)
        """
        im = ix + iy
        freq = np.bincount(im.ravel().astype(np.int64))
        hx = freq[self.rx] if len(freq) > self.rx else 0
        hy = freq[self.ry] if len(freq) > self.ry else 0
        ha = freq[self.ra] if len(freq) > self.ra else 0

        return hx, hy, ha

    def is_border(self, img):
        return img[:, 0].sum() > 0 or \
               img[:, -1].sum() > 0 or \
               img[0, :].sum() > 0 or \
               img[-1, :].sum() > 0