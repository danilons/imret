# coding: utf-8
import pandas as pd
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

    def class_id(self, name):
        class_names = self.class_names.get(name, name)
        return self.names.index(class_names)

    def get_name(self, class_id):
        return self.names[class_id]

    def color_from_id(self, class_id):
        return self[self.get_name(class_id)]

    def get_original_names(self, name):
        names = []
        for k, v in self.class_names.iteritems():
            if name == v:
                names.append(k)
        return names
