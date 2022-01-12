#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Protein-ligand docking rescoring with OnionNet-SFCT.

Author:
    Liangzhen Zheng - June 4, 2021

Contact:
   astrozheng@gmail.com
"""


xs2ele_dict = {
    0: "C",
    1: "C",
    2: "N",
    3: "N",
    4: "N",
    5: "N",
    6: "O",
    7: "O",
    8: "S",
    9: "P",
    10: "F",
    11: "Cl",
    12: "Br",
    13: "I",
    # 14: "Met",
}

RFSCORE_ELEMENTS = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']

element_dict_receptor = {
    0: "C",
    1: "N",
    2: "O",
    3: "S",
    4: "P",
    5: "F",
    6: "Cl",
    7: "Br",
    8: "I",
}

elements_ligand = ["H", "C", "O", "N", "P", "S", "Hal", "DU"]
elements_protein = ["H", "C", "O", "N", "P", "S", "Hal", "DU"]

# Define all residue types
all_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']


def get_residue(residue):
    if residue in all_residues:
        return residue
    else:
        return 'OTH'


# Define all element types
all_elements = ['H', 'C', 'O', 'N', 'P', 'S', 'Hal', 'DU']
all_heavy_elements = ['C', 'O', 'N', 'P', 'S', 'Hal', 'DU']
Hal = ['F', 'Cl', 'Br', 'I']


def get_elementtype(e):
    if e in all_elements:
        return e
    elif e in Hal:
        return 'Hal'
    else:
        return 'DU'


def get_protein_elementtype(e):
    if e in elements_protein:
        return e
    else:
        return "DU"


def get_ligand_elementtype(e):
    '''if e == "C.ar":
        return "CAR"  '''
    if e.split(".")[0] in elements_ligand:
        return e.split(".")[0]
    else:
        return "DU"

#  Bond
BONDTYPE2INT = {
    'SINGLE': 1,
    'DOUBLE': 2,
    'TRIPLE': 3,
    'AROMATIC': 4,
    'UNSPECIFIED': 5,
}

# Atomic number
ATOMIC_NUMBER = {
    'C': 12.,
    'N': 14.,
    'O': 16.,
    'P': 30.,
    'S': 32.,
    'F': 18.,
    'Cl': 35.,
    'Br': 70.,
    'I': 106.,
}

# copied from mdtraj atom radii
_ATOMIC_RADII = {'H': 0.120, 'He': 0.140, 'Li': 0.076, 'Be': 0.059,
                 'B': 0.192, 'C': 0.170, 'N': 0.155, 'O': 0.152,
                 'F': 0.147, 'Ne': 0.154, 'Na': 0.102, 'Mg': 0.086,
                 'Al': 0.184, 'Si': 0.210, 'P': 0.180, 'S': 0.180,
                 'Cl': 0.181, 'Ar': 0.188, 'K': 0.138, 'Ca': 0.114,
                 'Sc': 0.211, 'Ti': 0.200, 'V': 0.200, 'Cr': 0.200,
                 'Mn': 0.200, 'Fe': 0.200, 'Co': 0.200, 'Ni': 0.163,
                 'Cu': 0.140, 'Zn': 0.139, 'Ga': 0.187, 'Ge': 0.211,
                 'As': 0.185, 'Se': 0.190, 'Br': 0.185, 'Kr': 0.202,
                 'Rb': 0.303, 'Sr': 0.249, 'Y': 0.200, 'Zr': 0.200,
                 'Nb': 0.200, 'Mo': 0.200, 'Tc': 0.200, 'Ru': 0.200,
                 'Rh': 0.200, 'Pd': 0.163, 'Ag': 0.172, 'Cd': 0.158,
                 'In': 0.193, 'Sn': 0.217, 'Sb': 0.206, 'Te': 0.206,
                 'I': 0.198, 'Xe': 0.216, 'Cs': 0.167, 'Ba': 0.149,
                 'La': 0.200, 'Ce': 0.200, 'Pr': 0.200, 'Nd': 0.200,
                 'Pm': 0.200, 'Sm': 0.200, 'Eu': 0.200, 'Gd': 0.200,
                 'Tb': 0.200, 'Dy': 0.200, 'Ho': 0.200, 'Er': 0.200,
                 'Tm': 0.200, 'Yb': 0.200, 'Lu': 0.200, 'Hf': 0.200,
                 'Ta': 0.200, 'W': 0.200, 'Re': 0.200, 'Os': 0.200,
                 'Ir': 0.200, 'Pt': 0.175, 'Au': 0.166, 'Hg': 0.155,
                 'Tl': 0.196, 'Pb': 0.202, 'Bi': 0.207, 'Po': 0.197,
                 'At': 0.202, 'Rn': 0.220, 'Fr': 0.348, 'Ra': 0.283,
                 'Ac': 0.200, 'Th': 0.200, 'Pa': 0.200, 'U': 0.186,
                 'Np': 0.200, 'Pu': 0.200, 'Am': 0.200, 'Cm': 0.200,
                 'Bk': 0.200, 'Cf': 0.200, 'Es': 0.200, 'Fm': 0.200,
                 'Md': 0.200, 'No': 0.200, 'Lr': 0.200, 'Rf': 0.200,
                 'Db': 0.200, 'Sg': 0.200, 'Bh': 0.200, 'Hs': 0.200,
                 'Mt': 0.200, 'Ds': 0.200, 'Rg': 0.200, 'Cn': 0.200,
                 'Uut': 0.200, 'Fl': 0.200, 'Uup': 0.200, 'Lv': 0.200,
                 'Uus': 0.200, 'Uuo': 0.200}
