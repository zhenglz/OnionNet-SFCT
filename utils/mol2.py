# -*- coding: utf-8 -*-

"""Protein-ligand docking rescoring with OnionNet-SFCT.

Author:
    Liangzhen Zheng - June 4, 2021

Contact:
   astrozheng@gmail.com
"""

import sys

try:
    from rdkit import Chem
    from rdkit.Chem import rdFreeSASA
except:
    print("Rdkit not loaded")

import pandas as pd
import numpy as np
import mdtraj as mt
from utils.ligand import _get_bonds_by_distance
from utils.pdbqt import PdbqtParser
from utils.atomtype import RFSCORE_ELEMENTS, BONDTYPE2INT, ATOMIC_NUMBER


class LigandParser(object):
    """Parse the ligand with biopanda to obtain coordinates and elements.
    Parameters
    ----------
    ligand_fn : str,
        The input ligand file name.
    file_format: str,
        The input molecule file format.

    Methods
    -------
    Attributes
    ----------
    lig : a biopandas mol2 read object
    lig_data : a panda dataframe object holding the atom information
    coordinates : np.ndarray, shape = [ N, 3]
        The coordinates of the atoms in the ligand, N is the number of atoms.
    """

    def __init__(self, ligand_fn, file_format="pdb"):
        self.lig_file = ligand_fn
        self.file_format = file_format
        self.mol = None
        self.lig_ele = None
        self.lig_data = pd.DataFrame()
        self.coordinates_ = None
        self.atom_symbols = None
        self.atom_hybridization = None
        self.atom_in_aromatic = None
        self.atomic_number = None
        self.atom_in_ring = None
        self.sasa = None

    def _load_mol(self):
        if self.file_format == "pdb":
            self.mol = Chem.MolFromPDBFile(self.lig_file)
        elif self.file_format == "mol2":
            self.mol = Chem.MolFromMol2File(self.lig_file)
        elif self.file_format == "sdf":
            self.mol = Chem.SDMolSupplier(self.lig_file, sanitize=True)[0]
        else:
            self.mol = None

        if self.mol is None:
            '''import uuid
            mol2file = "/tmp/mols/{}.mol2".format(str(uuid.uuid4().hex))
            cmd = "obabel {} -O {}".format(self.lig_file, mol2file)
            job = sp.Popen(cmd, shell=True)
            job.communicate()

            self.lig_file = self.lig_file + ".sdf"
            cmd = "obabel {} -O {}".format(mol2file, self.lig_file)
            job = sp.Popen(cmd, shell=True)
            job.communicate()'''
            self.lig_file = self.lig_file + "_from_mol2.sdf"

            self.mol = Chem.SDMolSupplier(self.lig_file, sanitize=False)[0]

        return self.mol

    def _remove_hydrogens(self):
        if self.mol is None:
            self._load_mol()

        if self.mol is not None:
            try:
                self.mol = Chem.RemoveHs(self.mol)
            except:
                pass
        else:
            print("Warning: molecule processing error!!!")

        self.atoms = [x for x in self.mol.GetAtoms()]
        self.atom_number = len(self.atoms)

        return self.mol

    def _get_atom_symbol(self, ndx):
        if self.atom_symbols is None:
            self.atom_symbols = [''] * self.atom_number
            # self.atom_symbols = np.zeros(self.atom_number)

            for i in range(self.atom_number):
                # print("Atom %d ---> %s" %(i, self.atoms[i].GetSymbol()))
                self.atom_symbols[i] = self.atoms[i].GetSymbol()

        return self.atom_symbols[ndx]

    def _get_elements(self):
        self._get_atom_symbol(0)
        self.lig_ele = [x if x in RFSCORE_ELEMENTS else "DU" for x in self.atom_symbols]
        return self.lig_ele

    def _get_coordinates(self):
        """
        Get the coordinates in the pdb file given the ligand indices.
        Returns
        -------
        self : an instance of itself
        """
        conformer = self.mol.GetConformers()[0]
        self.coordinates_ = conformer.GetPositions()
        return self.coordinates_

    def _get_atom_hybridizations(self):

        if self.atom_hybridization is None:
            self.atom_hybridization = ['', ] * self.atom_number
            # self.atom_in_aromatic = np.zeros(self.atom_number)

            for i in range(self.atom_number):
                self.atom_hybridization[i] = str(self.atoms[i].GetHybridization())
                # print(self.atom_symbols[i], self.atom_hybridization[i])

            self.atom_hybridization = np.array(self.atom_hybridization)

        return self.atom_hybridization

    def _get_atom_in_aromatic(self):
        if self.atom_in_aromatic is None:

            self.atom_in_aromatic = np.zeros(self.atom_number)

            for i in range(self.atom_number):
                self.atom_in_aromatic[i] = self.atoms[i].GetIsAromatic()

        return self.atom_in_aromatic

    def _get_atomic_number(self):
        if self.atomic_number is None:
            self.atomic_number = np.zeros(self.atom_number)

            for i in range(self.atom_number):
                self.atomic_number[i] = self.atoms[i].GetAtomicNum()

        return self.atomic_number

    def _get_atom_in_ring(self):
        if self.atom_in_ring is None:
            self.atom_in_ring = np.zeros(self.atom_number)

            r = self.mol.GetRingInfo()
            atoms_in_ring = []
            for _r in r.AtomRings():
                for i in _r:
                    atoms_in_ring.append(i)

            for i in range(self.atom_number):
                if i in atoms_in_ring:
                    self.atom_in_ring[i] = 1

        return self.atom_in_ring

    def _get_atom_sasa(self):
        if self.sasa is None:
            self.radii = rdFreeSASA.classifyAtoms(self.mol)
            self.total_sasa = rdFreeSASA.CalcSASA(self.mol, self.radii)
            sasa_list = []
            for x in self.mol.GetAtoms():
                try:
                    sasa_list.append(float(x.GetProp('SASA')))
                except:
                    # print("atomic sasa", x)
                    sasa_list.append(0.0)
            # print(self.atom_number, len(sasa_list), sasa_list)
            self.sasa = np.array(sasa_list[:self.atom_number])

        return self.sasa

    def parse_mol(self):
        # load the molecule
        self._load_mol()
        # remove hydrogens
        self._remove_hydrogens()
        # get elements
        self._get_elements()
        # get coordinates
        self._get_coordinates()
        # is in aromatic
        self._get_atom_in_aromatic()
        # hybridization
        self._get_atom_hybridizations()
        # atomic number
        self._get_atomic_number()
        # atom in ring
        self._get_atom_in_ring()
        # sasa
        self._get_atom_sasa()

        self.lig_data['element'] = self.lig_ele

        _coord_str = ['x', 'y', 'z']
        for i in range(3):
            # unit angstrom
            self.lig_data[_coord_str[i]] = self.coordinates_[:, i]

        self.lig_data['is_aromatic'] = self.atom_in_aromatic
        self.lig_data['hybridization'] = self.atom_hybridization
        self.lig_data['atomic_number'] = self.atomic_number
        self.lig_data['atom_in_ring'] = self.atom_in_ring
        self.lig_data['sasa'] = self.sasa
        # print(self.lig_data)

        return self.lig_data

    def _atom_distance(self, coord1, coord2):

        return np.sqrt(np.mean(np.square(coord1 - coord2)))

    def get_bonds(self):
        self.bonds = []

        for bond in self.mol.GetBonds():
            b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            _distance = self._atom_distance(self.coordinates_[b], self.coordinates_[e])
            _type = str(bond.GetBondType())
            if _type in BONDTYPE2INT.keys():
                _type = BONDTYPE2INT[_type]
            else:
                _type = BONDTYPE2INT['UNSPECIFIED']

            if b < e:
                bond_data = [b, e, _type, _distance]
            else:
                bond_data = [e, b, _type, _distance]

            # print(bond_data)
            self.bonds.append(bond_data)

        return self.bonds


class LigandParser_Direct(LigandParser):
    def __init__(self, ligand_fn, file_format="mol2"):
        super().__init__(ligand_fn, file_format=file_format)
        self.mol_loaded_ = False

    def _read_atom_block(self):
        if self.file_format == "mol2":
            data_lines = []
            atom_block = False
            with open(self.lig_file) as lines:
                for l in lines:
                    if "@<TRIPOS>ATOM" in l:
                        atom_block = True
                    elif "@<TRIPOS>BOND" in l:
                        atom_block = False
                    else:
                        if atom_block and len(l.split()) and l.split()[5].split(".")[0] != "H":
                            data_lines.append(l.split())
                        else:
                            pass
        elif self.file_format in ['pdb', 'pdbqt']:
            with open(self.lig_file) as lines:
                data_lines = [x for x in lines if len(x.split()) and x.split()[0] in ['ATOM', 'HETATM']]
        else:
            data_lines = []

        return data_lines

    def _read_bond_block(self):
        data_lines = []
        atom_block = False
        with open(self.lig_file) as lines:
            for l in lines:
                if "@<TRIPOS>ATOM" in l:
                    atom_block = False
                elif "@<TRIPOS>BOND" in l:
                    atom_block = True
                elif "@<TRIPOS>SUBSTRUCTURE" in l:
                    atom_block = False
                else:
                    if atom_block and len(l.split()):
                        data_lines.append(l.split())
                    else:
                        pass

        return data_lines

    def _load_mol(self):
        lig_data = self._read_atom_block()

        ligand_df = pd.DataFrame([])

        if self.file_format == "mol2":
            ligand_df['atom_index'] = [str(x[0]) for x in lig_data]
            ligand_df['atom_name'] = [str(x[1]) for x in lig_data]
            ligand_df['x'] = [float(x[2]) for x in lig_data]
            ligand_df['y'] = [float(x[3]) for x in lig_data]
            ligand_df['z'] = [float(x[4]) for x in lig_data]
            ligand_df['atom_type'] = [str(x[5]) for x in lig_data]
            ligand_df['element'] = [str(x[5].split(".")[0]) for x in lig_data]
            ligand_df['resname'] = [str(x[7]) for x in lig_data]
            ligand_df['charge'] = [float(x[8]) for x in lig_data]
        elif self.file_format in ['pdb', 'pdbqt']:
            _parser = PdbqtParser(self.lig_file)
            ligand_df = _parser.to_dataframe()
            # atom_type
            # TO BE FIXED
        else:
            pass

        self.mol_loaded_ = True

        self.lig_infor = ligand_df.copy()
        self.atom_number = self.lig_infor.shape[0]

        return self

    def _get_elements(self):
        if not self.mol_loaded_:
            self._load_mol()

        self.lig_ele = [x if x in RFSCORE_ELEMENTS else "DU" for x in self.lig_infor['element'].values]
        return self.lig_ele

    def _get_coordinates(self):
        if not self.mol_loaded_:
            self._load_mol()

        self.coordinates_ = self.lig_infor[['x', 'y', 'z']].values
        return self.coordinates_

    def _get_atom_hybridizations(self):
        atom_hybridization = []

        for atom_type in self.lig_infor['atom_type'].values:
            # print("ATOM_TYPE", atom_type)
            if len(atom_type.split(".")) > 1:
                if atom_type.split(".")[1] == "3":
                    atom_hybridization.append("SP3")
                elif atom_type.split(".")[1] == "4":
                    atom_hybridization.append("SP3")
                elif atom_type.split(".")[1] == "co2":
                    atom_hybridization.append("SP2")
                elif atom_type.split(".")[1] == "2":
                    atom_hybridization.append("SP2")
                elif atom_type.split(".")[1] == "1":
                    atom_hybridization.append("SP1")
                elif atom_type.split(".")[1] == "am":
                    atom_hybridization.append("SP2")
                elif atom_type.split(".")[1] == "ar":
                    atom_hybridization.append("SP2")
                else:
                    atom_hybridization.append("UNKNOWN")
            else:
                atom_hybridization.append("SP3")

        self.atom_hybridization = np.array(atom_hybridization)

        return self

    def _get_atom_in_aromatic(self):
        atom_aromatic = []
        for atom_type in self.lig_infor['atom_type'].values:
            if len(atom_type.split(".")) == 1:
                atom_aromatic.append(0.0)
            else:
                if atom_type.split(".")[1] == "ar":
                    atom_aromatic.append(1.0)
                else:
                    atom_aromatic.append(0.0)
        self.atom_in_aromatic = np.array(atom_aromatic)

        return self.atom_in_aromatic

    def _get_atomic_number(self):
        atomic_numbers = []
        for _e in self.lig_infor['element'].values:
            if _e in ATOMIC_NUMBER.keys():
                atomic_numbers.append(ATOMIC_NUMBER[_e])
            else:
                atomic_numbers.append(ATOMIC_NUMBER['C'])

        self.atomic_number = np.array(atomic_numbers)

        return self.atomic_number

    def _get_atom_in_ring(self):
        self.atom_in_ring = np.zeros(self.atom_number)
        return self.atom_in_ring

    def _get_atom_sasa(self):
        _xyz = self.coordinates_.reshape((1, self.atom_number, 3)) * 0.1
        top = mt.Topology()
        _df = pd.DataFrame()
        _df['serial'] = np.arange(self.atom_number)
        _df['name'] = self.lig_infor['atom_name'].values
        _df['element'] = [x if x != "DU" else "C" for x in self.lig_ele]
        _df['resSeq'] = [1, ] * self.atom_number
        _df['resName'] = ['LIG'] * self.atom_number
        _df['chainID'] = [0, ] * self.atom_number
        top = top.from_dataframe(_df, None)
        # print("N atoms ", top._numAtoms)

        traj = mt.Trajectory(_xyz, top)
        # traj.xyz(_xyz)

        self.sasa = mt.shrake_rupley(traj)[0] * 100.0

        return self.sasa

    def parse_mol(self):
        # load the molecule
        self._load_mol()
        # remove hydrogens
        # self._remove_hydrogens()
        # get elements
        self._get_elements()
        # get coordinates
        self._get_coordinates()
        # is in aromatic
        self._get_atom_in_aromatic()
        # hybridization
        self._get_atom_hybridizations()
        # atomic number
        self._get_atomic_number()
        # atom in ring
        self._get_atom_in_ring()
        # sasa
        self._get_atom_sasa()

        self.lig_data['element'] = self.lig_ele

        _coord_str = ['x', 'y', 'z']
        for i in range(3):
            # unit angstrom
            self.lig_data[_coord_str[i]] = self.coordinates_[:, i]

        self.lig_data['is_aromatic'] = self.atom_in_aromatic
        self.lig_data['hybridization'] = self.atom_hybridization
        self.lig_data['atomic_number'] = self.atomic_number
        self.lig_data['atom_in_ring'] = self.atom_in_ring
        self.lig_data['sasa'] = self.sasa
        # print(self.lig_data)

        return self.lig_data

    def _get_bond_type(self, type_str="1"):
        if type_str == "1":
            return "SINGLE"
        elif type_str == "2":
            return "DOUBLE"
        elif type_str == "3":
            return "TRIPLE"
        elif type_str == "ar":
            return "AROMATIC"
        elif type_str == "am":
            return "DOUBLE"
        else:
            return "UNSPECIFIED"

    def get_bonds(self):
        self.bonds = []
        bond_lines = self._read_bond_block()

        for bond in bond_lines:

            b, e = bond[1], bond[2]

            try:
                b_ndx = self.lig_infor[self.lig_infor['atom_index'] == b].index.values[0]
                e_ndx = self.lig_infor[self.lig_infor['atom_index'] == e].index.values[0]
                # print("BOND ", b, e, b_ndx, e_ndx)
            except:
                # print("atoms including hydrogens", b, e)
                continue

            _distance = self._atom_distance(self.coordinates_[b_ndx], self.coordinates_[e_ndx])

            _type = self._get_bond_type(bond[-1])
            if _type in BONDTYPE2INT.keys():
                _type = BONDTYPE2INT[_type]
            else:
                _type = BONDTYPE2INT['UNSPECIFIED']

            if b_ndx < e_ndx:
                bond_data = [b_ndx, e_ndx, _type, _distance]
            else:
                bond_data = [e_ndx, b_ndx, _type, _distance]

            # print(bond_data)
            if _distance < 1.3:
                self.bonds.append(bond_data)
            else:
                print("Strange bond length {:.3f} for {} and {}".format(_distance, b, e))

        return self.bonds

    def get_bonds_by_distance(self, scale=0.9):

        self.bonds = _get_bonds_by_distance(self.coordinates_,
                                            self.atom_number,
                                            self.lig_ele)

        return self.bonds


if __name__ == "__main__":
    filename = sys.argv[1]

    parser = LigandParser_Direct(ligand_fn=filename, file_format="mol2")

    parser.parse_mol()
    print(parser.lig_infor.head())
    # print(parser.lig_data)

    print("Bonds 1")
    b1 = parser.get_bonds_by_distance(scale=0.7)
    print(b1, len(b1))
    print("Bonds 2")
    b2 = parser.get_bonds()
    print(b2, len(b2))
