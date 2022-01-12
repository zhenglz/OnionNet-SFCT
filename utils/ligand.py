#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Protein-ligand docking rescoring with OnionNet-SFCT.

Author:
    Liangzhen Zheng - June 4, 2021

Contact:
   astrozheng@gmail.com
"""


import mdtraj as mt
from biopandas.mol2 import PandasMol2
import os
import numpy as np
import subprocess as sp

try:
    from rdkit import Chem
except ImportError:
    print('Rdkit not imported...')

from utils.atomtype import BONDTYPE2INT, _ATOMIC_RADII
from utils.distance import fast_distance_pairs
from utils.bond import BondType


class Molecule(object):
    """Small molecule parser object with Rdkit package.
    Parameters
    ----------
    in_format : str, default = 'smile'
        Input information (file) format.
        Options: smile, pdb, sdf, mol2, mol
    Attributes
    ----------
    molecule_ : rdkit.Chem.Molecule object
    mol_file : str
        The input file name or Smile string
    converter_ : dict, dict of rdkit.Chem.MolFrom** methods
        The file loading method dictionary. The keys are:
        pdb, sdf, mol2, mol, smile
    """

    def __init__(self, in_format="smile"):

        self.format = in_format
        self.molecule_ = None
        self.mol_file = None
        self.converter_ = None
        self.mol_converter()

    def mol_converter(self):
        """The converter methods are stored in a dictionary.
        Returns
        -------
        self : return an instance of itself
        """
        try:
            from rdkit import Chem
        except:
            print("INFO: rdkit is not imported")

        self.converter_ = {
            "pdb": Chem.MolFromPDBFile,
            "mol2": Chem.MolFromMol2File,
            "mol": Chem.MolFromMolFile,
            "smile": Chem.MolFromSmiles,
            "sdf": Chem.MolFromMolBlock,
            "pdbqt": self.babel_converter,
        }

        return self

    def babel_converter(self, mol_file, output):
        if os.path.exists(mol_file):
            try:
                cmd = 'obabel %s -O %s > /dev/null' % (mol_file, output)
                job = sp.Popen(cmd, shell=True)
                job.communicate()

                self.molecule_ = self.converter_['pdb']()
                return self.molecule_
            except:
                return None
        else:
            print("No such input for converting: ", mol_file)

        return None

    def load_molecule(self, mol_file):
        """Load a molecule to have a rdkit.Chem.Molecule object
        Parameters
        ----------
        mol_file : str
            The input file name or SMILE string
        Returns
        -------
        molecule : rdkit.Chem.Molecule object
            The molecule object
        """

        self.mol_file = mol_file
        if not os.path.exists(self.mol_file):
            print("Molecule file not exists. ")
            return None

        if self.format not in ["mol2", "mol", "pdb", "sdf", "pdbqt"]:
            print("File format is not correct. ")
            return None
        else:
            try:
                self.molecule_ = self.converter_[self.format](self.mol_file, sanitize=True,)
            except RuntimeError:
                return None

            return self.molecule_


def _get_bonds_by_distance(coordinates_, atom_number, lig_ele, scale=0.9):
    bonds = []

    assert coordinates_.shape[0] == atom_number

    distance_matrix = fast_distance_pairs(coordinates_, coordinates_).reshape((atom_number, -1))

    for i in range(atom_number):
        for j in range(atom_number):
            if i < j:
                _d = distance_matrix[i, j]
                _element_i = lig_ele[i]
                _element_j = lig_ele[j]

                try:
                    _radii_i = _ATOMIC_RADII[_element_i]
                except:
                    _radii_i = _ATOMIC_RADII["C"]

                try:
                    _radii_j = _ATOMIC_RADII[_element_j]
                except:
                    _radii_j = _ATOMIC_RADII["C"]

                if _d < scale * (_radii_i + _radii_j) * 10.0:
                    bondtype = BondType(_element_i, _element_j, _d)
                    _type = bondtype.get_bond_type()
                    if _type is None:
                        continue
                    else:
                        # print(_element_i, _element_j, i, j, _d, _type, (_radii_i + _radii_j)*10.0)
                        bonds.append([i, j, BONDTYPE2INT[_type], _d])

    return bonds
