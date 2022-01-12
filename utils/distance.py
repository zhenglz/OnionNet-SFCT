#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Protein-ligand docking rescoring with OnionNet-SFCT.

Author:
    Liangzhen Zheng - June 4, 2021

Contact:
   astrozheng@gmail.com
"""

import numpy as np
import mdtraj as mt
import itertools
from utils.atomtype import RFSCORE_ELEMENTS


def distance2counts(megadata):
    d, c, charge_pairs, distance_mode, distance_delta = megadata

    if charge_pairs is None:
        return np.sum((np.array(d) <= c) * 1.0)
    else:
        atompair_in_dist_range = ((np.array(d) <= c) & (np.array(d) < c - distance_delta)) * 1.0

        if distance_mode == 'cutoff':
            return np.multiply(atompair_in_dist_range, np.array(charge_pairs) / c)
        else:
            return np.multiply(atompair_in_dist_range, np.divide(charge_pairs, d))


def dist2count_simple(distances, cutoff):
    return (distances <= cutoff) * 1.0


def fast_distance_pairs(coords_pro, coords_lig):
    return np.array([np.linalg.norm(coord_pro - coords_lig, axis=-1) for coord_pro in coords_pro]).ravel()


def distance_pairs_mdtraj(coord_pro, coord_lig):
    xyz = np.concatenate((coord_pro, coord_lig), axis=0)
    # print(xyz.shape)
    # mdtraj use nanometer for coordinations,
    # convert angstrom to nanometer
    xyz = xyz.reshape((1, -1, 3)) * 0.1
    # for the xyz, the sample is (N_frames, N_atoms, N_dim)
    # N_frames, it is usually 1 for a normal single-molecule PDB
    # N_atoms is the number of atoms in the pdb file
    # N_dims is the dimension of coordinates, 3 here for x, y and z

    # create a mdtraj Trajectory object,Topology object could be ignored.
    traj = mt.Trajectory(xyz=xyz, topology=None)
    # create a list of atom-pairs from atom index of protein and ligand
    atom_pairs = itertools.product(np.arange(coord_pro.shape[0]),
                                   np.arange(coord_pro.shape[0], coord_pro.shape[0] + coord_lig.shape[0]))

    # convert the distance to angstrom from nanometer.
    # Actually we could just leave it as angstrom from the beginning for faster calculation,
    # but it is more reasonable to do it in order to aligning with mdtraj-style calculation.
    dist = mt.compute_distances(traj, atom_pairs)[0] * 10.0

    return dist


def residue_min_distance(coord_pro, coord_lig, residue_names, receptor_elements, ligand_elements, use_mean=False):
    # combine same residues in the residue_names list
    uniq_res = []
    for _residue in residue_names:
        if _residue not in uniq_res:
            uniq_res.append(_residue)

    # print(coord_pro.shape, len(residue_names))

    # assert coord_pro.shape[0] == len(residue_names)
    assert coord_pro.shape[0] == len(receptor_elements)
    assert coord_lig.shape[0] == len(ligand_elements)

    _ligand_indices = np.array([x for x in range(len(ligand_elements)) if ligand_elements[x] != "H"])

    results = np.zeros((len(uniq_res), _ligand_indices.shape[0]))

    for i, resid in enumerate(uniq_res):
        _receptor_indices = np.array([x for x in range(coord_pro.shape[0])
                                      if (residue_names[x] == resid and
                                          receptor_elements[x] != "H")])
        # print(_receptor_indices, _ligand_indices)

        _distances = fast_distance_pairs(coord_pro[_receptor_indices], coord_lig[_ligand_indices])
        _distances = _distances.reshape((-1, _ligand_indices.shape[0]))

        if use_mean:
            # find the min distance
            _min_dist = np.mean(_distances, axis=0)
        else:
            _min_dist = np.min(_distances, axis=0)
        # print(_min_dist.shape, _ligand_indices.shape)
        results[i] = _min_dist
        # if np.min(_min_dist) < 3.5:
        #    print(resid)

    return (results, uniq_res, [x for x in ligand_elements if x != "H"])


def residue_mean_distance(coord_pro, coord_lig, residue_names, receptor_elements, ligand_elements):
    return residue_min_distance(coord_pro, coord_lig, residue_names, receptor_elements, ligand_elements, use_mean=True)


def atom_atom_contacts(coord_pro, coord_lig, receptor_elements,
                       ligand_elements, cutoffs=[3.0, 4.0, 5.0]):
    assert coord_pro.shape[0] == len(receptor_elements)
    assert coord_lig.shape[0] == len(ligand_elements)

    _ligand_indices = np.array([x for x in range(len(ligand_elements)) if ligand_elements[x] != "H"])

    _distance_matrix = fast_distance_pairs(coord_lig, coord_pro).reshape((coord_lig.shape[0], coord_pro.shape[0]))

    results = np.zeros((coord_lig.shape[0], len(RFSCORE_ELEMENTS) * len(cutoffs)))
    for i in range(_ligand_indices.shape[0]):
        # _ligand_coords = coord_lig[_ligand_indices[i]].reshape((1, 3))
        # _receptor_coords =
        _distances_i = _distance_matrix[_ligand_indices[i]]
        _sum_contacts = np.zeros((len(RFSCORE_ELEMENTS), len(cutoffs)))

        for j, element in enumerate(RFSCORE_ELEMENTS):
            _receptor_indices = np.array([x for x in range(coord_pro.shape[0]) if receptor_elements[x] == element])
            if _receptor_indices.shape[0] > 0:
                _distances_i_ele = _distances_i[_receptor_indices]

                for k, _cutoff in enumerate(cutoffs):
                    _sum = np.sum(dist2count_simple(_distances_i_ele, _cutoff))
                    _sum_contacts[j, k] = _sum
            else:
                pass

        results[i] = _sum_contacts.ravel() / 15.0

    return results


if __name__ == "__main__":
    from utils.pdbqt import PdbqtParser
    from utils.mol2 import LigandParser
    import sys, os

    receptor = PdbqtParser(sys.argv[1])
    receptor.parse_pdbqt()
    rec_coords = receptor.df[['x', 'y', 'z']].values
    rec_elements = list(receptor.df['element'].values)

    ligand = LigandParser(sys.argv[2], file_format='mol2')
    ligand.parse_mol()
    lig_coords = ligand.lig_data[['x', 'y', 'z']].values
    lig_elements = list(ligand.lig_data['element'].values)

    r = atom_atom_contacts(rec_coords, lig_coords, rec_elements, lig_elements)
    print(1 - r / 10.)
    print(r.shape)
