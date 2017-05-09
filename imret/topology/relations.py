# coding: utf-8
import collections


class Relation(object):

    _longname = {'DC': 'Disconnected',
                 'EC': 'Externally connected',
                 'PO': 'Partially Overlapping',
                 'EQ': 'Equal',
                 'TPP': 'Tangential proper part',
                 'TPPi': 'Tangential proper part inverse',
                 'NTPP': 'Non-Tangential proper part',
                 'NTPPi': 'Non-Tangential proper part inverse',
                 'UNK': 'Unknown'}

    def __init__(self, scope):
        self.scope = scope
        self.relation = collections.OrderedDict()
        self.relation['rcc8'] = scope
        self.relation['rcc5'] = scope

    def __repr__(self):
        return "RCC relation {}: {}".format(self.scope, self._longname[self.scope])

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return self.relation[item]

    @classmethod
    def get_name(cls, relation):
        return cls._longname[relation]

    @classmethod
    def get_preffix(cls):
        return cls._longname.keys()