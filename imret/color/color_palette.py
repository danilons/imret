# coding: utf-8
import pandas as pd
import numpy as np
from .palette import PALETTE


class ColorPalette:
    def __init__(self, name_conversion):
        self.palette = PALETTE
        self.class_names = pd.read_csv(name_conversion).set_index('Class')['Name'].to_dict()
        self.names = ['__background__'] + sorted(set(self.class_names.values()))

    def __getitem__(self, name):
        class_names = self.class_names.get(name, name)
        index = self.names.index(class_names)
        return self.palette[index, :]

    def __contains__(self, item):
        return item in self.class_names

    def class_id(self, name):
        class_names = self.class_names.get(name, name)
        return self.names.index(class_names)

    def get_name(self, class_id):
        return self.names[class_id]

    def color_from_id(self, class_id):
        return self[self.get_name(class_id)]

    def get_original_names(self, name):
        names = []
        for k, v in self.class_names.items():
            if name == v:
                names.append(k)
        return names

    def save(self, file_name):
        with open(file_name, 'w') as fp:
            for name in self.names:
                class_id = self.class_id(name)
                fp.write("{}: \t {}\n".format(class_id, name))

    def save_colormap(self, file_name):
        with open(file_name, 'w') as fp:
            for name in self.names:
                r, g, b = self[name]
                fp.write("{} {} {} \n".format(r, g, b))