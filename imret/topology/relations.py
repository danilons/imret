# coding: utf-8
import cv2
import numpy as np
import itertools
from .rcc import RCC

detector = RCC()


def topology_relation(shape, objects):
    if len(objects) == 0:
        return []

    contours = objects.items()
    if len(objects) == 1:
        return [(contours, contours, 'EQ')]

    def draw(shape, contour, value):
        img = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(img, [contour], -1, value, -1)
        return img

    relations = []
    images = {}
    keys = set()
    for ((obj1, c1), (obj2, c2)) in itertools.combinations(contours, 2):
        key = '{}:{}'.format(obj1, obj2)
        if key in keys:
            continue

        x1, y1, w1, h1 = cv2.boundingRect(c1)
        x11, y11, x12, y12 = x1, y1, x1 + w1, y1 + h1
        x11, x12 = sorted([x11, x12])
        y11, y12 = sorted([y11, y12])
        contour1 = np.array([[x11, y11], [x11, y12], [x12, y12], [x12, y11]])

        imx = images.get(obj1, None)
        if imx is None:
            imx = draw(shape, contour1, value=detector.x_value)
            images[obj1] = imx

        x2, y2, w2, h2 = cv2.boundingRect(c2)
        x21, y21, x22, y22 = x2, y2, x2 + w2, y2 + h2
        x21, x22 = sorted([x21, x22])
        y21, y22 = sorted([y21, y22])
        contour2 = np.array([[x21, y21], [x21, y22], [x22, y22], [x22, y21]])

        imy = images.get(obj2, None)
        if imy is None:
            imy = draw(shape, contour2, value=detector.y_value)
            images[obj2] = imy

        relation = detector.detect(imx, imy)
        relations.append({'objects': (obj1, obj2),
                          'contours': (contour1, contour2),
                          'relation': relation.scope})
        keys.add(key)

    return relations