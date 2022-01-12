#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Protein-ligand docking rescoring with OnionNet-SFCT.

Author:
    Liangzhen Zheng - June 4, 2021

Contact:
   astrozheng@gmail.com
"""

# https://www.transtutors.com/questions/8-100-the-bond-lengths-of-carbon-carbon-carbon-nitrogen-carbon-oxygen-and-nitrogen-n-1525201.htm


_SINGLE_BOND_ELEMENTS = ['H', 'Cl', 'F', 'Br', 'I', 'P']


class BondType(object):

    def __init__(self, element1, element2, distance):
        self.element1 = element1
        self.element2 = element2

        self.distance = distance

    def _single_bond(self):

        if self.element1 in _SINGLE_BOND_ELEMENTS or \
                self.element2 in _SINGLE_BOND_ELEMENTS:
            return "SINGLE"
        else:
            return None

    def _aromatic_bond(self):

        return None

    def _mix_types(self):

        if self.element1 == "C" and self.element2 == "C":
            if self.distance > 1.50 and self.distance <= 1.8:
                return "SINGLE"
            elif self.distance > 1.8:
                return None
            elif self.distance <= 1.5 and self.distance > 1.3:
                return "DOUBLE"
            elif self.distance <= 1.3:
                return "TRIPLE"
            else:
                return None

        if (self.element1 == "O" and self.element2 == "C") or \
                (self.element1 == "C" and self.element2 == "O"):
            if self.distance > 1.40 and self.distance <= 1.8:
                return "SINGLE"
            elif self.distance > 1.8:
                return None
            elif self.distance <= 1.4 and self.distance > 1.20:
                return "DOUBLE"
            elif self.distance <= 1.2:
                return "TRIPLE"
            else:
                return None

        if (self.element1 == "N" and self.element2 == "C") or \
                (self.element1 == "C" and self.element2 == "N"):
            if self.distance > 1.45 and self.distance <= 1.8:
                return "SINGLE"
            elif self.distance > 1.8:
                return None
            elif self.distance <= 1.45 and self.distance > 1.20:
                return "DOUBLE"
            elif self.distance <= 1.2:
                return "TRIPLE"
            else:
                return None

        if self.element1 == "N" and self.element2 == "N":
            if self.distance > 1.35 and self.distance <= 1.6:
                return "SINGLE"
            elif self.distance > 16:
                return None
            elif self.distance <= 1.35 and self.distance > 1.20:
                return "DOUBLE"
            elif self.distance <= 1.2:
                return "TRIPLE"
            else:
                return None

        if (self.element1 == "N" and self.element2 == "O") or \
                (self.element1 == "O" and self.element2 == "N"):
            if self.distance > 1.45:
                return None
            elif self.distance <= 1.45 and self.distance > 1.3:
                return "SINGLE"
            elif self.distance <= 1.3:
                return "DOUBLE"
            else:
                return None

        if self.element1 == "S" and self.element2 == "S":
            if self.distance <= 2.1:
                return "SINGLE"
            else:
                return None

        if (self.element1 == "O" and self.element2 == "S") or \
                (self.element2 == "O" and self.element1 == "S"):
            if self.distance > 1.6:
                return None
            elif self.distance <= 1.6 and self.distance > 1.48:
                return "SINGLE"
            elif self.distance <= 1.48:
                return "DOUBLE"
            else:
                return None

        if (self.element1 == "C" and self.element2 == "S") or \
                (self.element2 == "C" and self.element1 == "S"):
            if self.distance > 1.8:
                return None
            elif self.distance <= 1.8:
                return "SINGLE"
            else:
                return None

        if (self.element1 == "N" and self.element2 == "S") or \
                (self.element2 == "N" and self.element1 == "S"):
            if self.distance > 1.8:
                return None
            elif self.distance <= 1.8:
                return "SINGLE"
            else:
                return None

    def get_bond_type(self):

        _bond_type = self._single_bond()
        if _bond_type is not None:
            return _bond_type

        _bond_type = self._mix_types()
        if _bond_type is not None:
            return _bond_type

        if self.distance < 2.0 and _bond_type is not None:
            return "UNSPECIFIED"
        else:
            return None

