#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Protein-ligand docking rescoring with OnionNet-SFCT.

Author:
    Liangzhen Zheng - June 4, 2021

Contact:
   astrozheng@gmail.com
"""

import os
import pandas as pd
from utils.atomtype import all_residues


class PdbqtParser(object):
    """Parse PDBQT file.
    Parameters
    ----------
    pdbqt_fn : str, or list
        Input pdbqt file. Or the line contents of the pdbqt file.
    Examples
    --------
    >>> pdbqt = PdbqtParser("pdb.pdbqt")
    >>> df = pdbqt.to_dataframe()
    >>> df.values
    >>> df.columns
    >>> df['serial']
    """

    def __init__(self, pdbqt_fn=None):
        self.fn = pdbqt_fn
        self.df = None
        self.coordinates_ = None
        self.rec_ele = None

    def _get_atom_lines(self):
        if isinstance(self.fn, list) and len(self.fn):
            return [x for x in self.fn if len(x.split()) > 5 and x.split()[0] == "ATOM"]

        if not os.path.exists(self.fn):
            print("INFO: No such pdbqt ")
            return []

        with open(self.fn) as lines:
            # lines = [x for x in lines if len(x.split()) > 5 and x[21] == "A"]
            lines = [x for x in lines if len(x.split()) > 5 and x.split()[0] == "ATOM"]
        return lines

    def _element_determinator(self, element, atomname):

        if len(element) == 1:
            if element == "A":
                return "C"
            elif element.upper() not in ['C', 'H', 'N', 'O', 'P', 'S', 'K', 'I', 'F']:
                print("INFO: find unusal element ", element, "and it will be replace by %s" % atomname[0].upper())
                return atomname[0].upper()
            else:
                return element.upper()
        elif len(element) == 2:
            e = element.upper()
            if e in ['BR', 'CL', 'MG', 'ZN']:
                return e
            else:
                return e[0]
        else:
            return "NA"

    def _residue_name_fix(self, resname):

        if resname in all_residues:
            return resname
        elif resname in ['HIE', 'HID', 'HIP']:
            return "HIS"
        elif resname in ['MES', 'MER', ]:
            return "MET"
        elif resname in ['CYX', 'CXS', ]:
            return "CYS"
        else:
            print("warning: strange resname {}".format(resname))
            return "OTH"

    def _parse_atom_line(self, line):
        try:
            _atom_id = int(line[6:11].strip())
        except:
            _atom_id = 0
            print("Warning: ", line)
        _atom_name = line[12:16].strip()
        _chainid = line[21]
        _res_name = self._residue_name_fix(line[17:20].strip())

        try:
            _res_id = int(line[22:26].strip())
        except ValueError:
            _res_id = 0
            print("warning: strange resid ")
            print(line)
            _res_id = -1
            # return None
        # coordination in unit: angstrom

        try:
            _x = float(line[30:38].strip())
            _y = float(line[38:46].strip())
            _z = float(line[46:54].strip())
        except:
            print("warning: strange pdbqt line")
            #print(line)
            _x = float(line.split()[5])
            _y = float(line.split()[6])
            _z = float(line.split()[7])

            #print(_x, _y, _z)

        _element = line[13]

        # _element = self._element_determinator(line[76:79].strip(), _atom_name)
        try:
            _charge = float(line[70:76].strip())
        except ValueError:
            _charge = 0.0

        return [_atom_id, _atom_name, _chainid, _res_name,
                _res_id, _x, _y, _z, _element, _charge]

    def to_dataframe(self):
        atom_lines = self._get_atom_lines()
        if atom_lines is None or len(atom_lines) == 0:
            return None
        else:
            _data = []
            for line in atom_lines:
                _items = self._parse_atom_line(line)
                if _items is not None:
                    _data.append(_items)
                else:
                    pass

            _df = pd.DataFrame(_data, columns=['serial', 'name', 'chain', 'resName',
                                               'resSeq', 'x', 'y', 'z', 'element', 'charge'])

            # self.df = _df[_df['chain'] == "A"]
            # print(_df)
            self.df = _df.copy()

            self.coordinates_ = _df[['x', 'y', 'z']].values
            # print(self.df['resSeq'].values)
            return self.df

    def parse_pdbqt(self):
        if self.df is None:
            self.to_dataframe()

        return self.df
