#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Protein-ligand docking rescoring with OnionNet-SFCT.

Author:
    Liangzhen Zheng - June 4, 2021

Contact:
   astrozheng@gmail.com
"""


import sys, os
import numpy as np
import itertools
import uuid
import pickle
import subprocess as sp
import argparse
from collections import OrderedDict
from biopandas.mol2 import split_multimol2
from sklearn.ensemble import AdaBoostRegressor

from utils.parallel import ParallelSim
from utils.mol2 import LigandParser_Direct
from utils.pdbqt import PdbqtParser
from utils.atomtype import all_residues, get_elementtype, all_heavy_elements
from utils.distance import residue_min_distance, dist2count_simple


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def gen_rabind_features(receptor_ha_coords_, ligand_ha_coords_,
                        receptor_resnames_, receptor_ha_elements_,
                        ligand_ha_elements_, cutoffs):
    # get mini distances
    _min_distances, _residues, _non_hydrogen_elements = \
        residue_min_distance(receptor_ha_coords_,
                             ligand_ha_coords_,
                             receptor_resnames_,
                             receptor_ha_elements_,
                             ligand_ha_elements_,
                             False)
    # print("Min distance", _min_distances, _min_distances.shape)

    # Types of the residue and the atom
    # new_residue = list(map(get_residue, cplx.residues))
    new_residue = [x.split("_")[0] for x in _residues]
    new_lig = list(map(get_elementtype, _non_hydrogen_elements))

    # residue-atom pairs
    residues_lig_atoms_combines = ["_".join(x) for x in \
                                   list(itertools.product(all_residues, all_heavy_elements))]

    # calculate the number of contacts in different shells
    counts = []
    onion_counts = []
    final_results = OrderedDict()

    for i, cutoff in enumerate(cutoffs):
        counts_ = dist2count_simple(_min_distances, cutoff)  # .ravel()
        counts.append(counts_.ravel())
        # print(len(onion_counts), counts_.shape, )
        if i == 0:
            onion_counts.append(list(counts_.ravel()))
        else:
            onion_counts.append(list(counts_.ravel() - counts[-1]))

        for j, _key in enumerate(residues_lig_atoms_combines):
            # print("COMBINE SIZE", len(residues_lig_atoms_combines))
            _new_key = _key + "_" + str(i)
            final_results[_new_key] = 0.0

        for j in range(len(new_residue)):
            for k in range(len(new_lig)):
                _new_key = new_residue[j] + "_" + new_lig[k] + "_" + str(i)
                if _new_key in final_results.keys():
                    final_results[_new_key] += counts_[j, k]
                else:
                    print("warning: unseen key in contact counts {}".format(_new_key))

    X = np.array(list(np.array(list(final_results.values())).ravel())).reshape((1, -1))
    return X


def score_pose(receptor_obj: PdbqtParser, ligand_obj: LigandParser_Direct,
               method: str ='sfct',
               model_obj: AdaBoostRegressor = None, index: int = 0) -> (float, int):
    if method == "onionnet2-rf":
        cutoffs = [1.5, 3.0, 4.5, 6, 7.5, 9, 10.5, 12.0]
    elif method == "sfct":
        cutoffs = [1.5, 3.0, 4.5, 6, 7.5, 9, 10.5, 12.0, 13.5, 15, 16.5, 18.0, 19.5, 21.0]
    else:
        cutoffs = None

    # unit angstrom
    rec_ele = receptor_obj.df['element'].values
    indices = (rec_ele != "H")
    rec_ele = rec_ele[indices]
    rec_xyz = receptor_obj.df[['x', 'y', 'z']].values[indices]
    # rec_res = receptor_obj.df['resName'].values[indices]
    # print("Before correct: ", rec_res)
    rec_res_correct = ["_".join([receptor_obj.df['resName'].values[x],
                                 str(receptor_obj.df['resSeq'].values[x]),
                                 receptor_obj.df['chain'].values[x]])
                       for x in range(receptor_obj.df.shape[0])
                       if receptor_obj.df['element'].values[x] != "H"]
    # print("After correct: ", rec_res_correct)
    assert rec_ele.shape[0] == rec_xyz.shape[0]
    # print("Receptor XYZ shape", rec_xyz.shape, rec_xyz[:5])

    lig_ele = ligand_obj.lig_data['element'].values
    indices = (lig_ele != "H")
    lig_ele = lig_ele[indices]
    lig_xyz = ligand_obj.lig_data[['x', 'y', 'z']].values[indices]
    assert lig_xyz.shape[0] == lig_ele.shape[0]
    # print("Ligand XYZ shape", lig_xyz.shape, lig_xyz[:5])

    # fix here
    X = gen_rabind_features(rec_xyz, lig_xyz, rec_res_correct, rec_ele, lig_ele, cutoffs)
    # print(np.sum(X))
    return X
    #if method == "rabind-cnn":
    #    X = X.reshape((1, 1, 14, 21 * 7))

    #score = model_obj.predict(X).ravel()[0]

    #return (score, index)


def parse_multimol2(filename, reference=None):
    mol2_object_dict = {}
    rmsd = {}
    mol2_code_unique = []
    mol2_code_list = []

    for counter, mol2 in enumerate(split_multimol2(filename)):
        mol2_code, mol2_lines = mol2
        mol2_code = mol2_code.split("/")[-1]

        if mol2_code in mol2_code_unique:
            mol2_code = "pose_{}".format(counter)

        #print("INFO: #%d Ligand ID code " % counter, mol2_code)
        mol2_code_list.append(mol2_code)

        fn = "{}_{}_{}.mol2".format(filename, str(uuid.uuid4().hex)[:6], mol2_code)
        with open(fn, 'w') as tofile:
            for line in mol2_lines:
                tofile.write(line)
        tofile.close()

        lig = LigandParser_Direct(fn)
        lig.parse_mol()
        mol2_object_dict[mol2_code] = lig

        if reference is not None and os.path.exists(reference):
            _rmsd = obrms_RMSD(reference, fn)
            rmsd[mol2_code] = _rmsd

        os.remove(fn)

        mol2_code_unique.append(mol2_code)

    return mol2_object_dict, rmsd, mol2_code_list


def obrms_RMSD(ref, query):
    os.makedirs('/tmp/rmsd/', exist_ok=True)
    rmsd_out = '/tmp/rmsd/rmsd.{}'.format(str(uuid.uuid4().hex))
    cmd = "obrms {} {} > {}".format(ref, query, rmsd_out)

    # run job
    job = sp.Popen(cmd, shell=True)
    job.communicate()

    with open(rmsd_out) as lines:
        try:
            rmsd = float([x.split()[-1] for x in lines][0])
        except:
            rmsd = 99.99

    try:
        os.remove(rmsd_out)
    except:
        pass

    return rmsd


def rescoring(receptor_file, ligand_file, output, model_pkl, method='rabind-rf',
              reference="ref.mol2", pose_scores=None, weight=1.0,
              ncpus=1):
    #score_model_object = pickle.load(open(model_pkl, 'rb'))
    score_model_object = None

    receptor_obj = PdbqtParser(receptor_file)
    receptor_obj.parse_pdbqt()

    mol2_objects, rmsd_values, mol2_code_list = parse_multimol2(ligand_file, reference)

    '''tofile = open(output, "w")
    if os.path.exists(reference):
        tofile.write("# name pose_index origin_score combined_score sfct RMSD(Angstrom)\n")
    else:
        tofile.write("# name pose_index origin_score combined_score sfct\n")
    '''
    #print("Original Scores: ", pose_scores)

    ml_scores = []
    combined_scores = []
    if ncpus <= 1:
        for i, mol_id in enumerate(mol2_code_list):
            ligand_obj = mol2_objects[mol_id]
            #score, _ = score_pose(receptor_obj, ligand_obj, method, score_model_object)
            X = score_pose(receptor_obj, ligand_obj, method, score_model_object)
            np.savetxt(output, X, fmt="%.1f")     
            '''ml_scores.append(score)

            if pose_scores is not None:
                score = score * weight + pose_scores[i] * (1.0 - weight)

            combined_scores.append(score)
    else:
        print("Scoring with multiple CPUs: ", ncpus)
        paral = ParallelSim(ncpus)
        for i, mol_id in enumerate(mol2_code_list):
            ligand_obj = mol2_objects[mol_id]

            paral.add(score_pose, (receptor_obj, ligand_obj, method, score_model_object, i))

        paral.run()

        results = paral.get_results()
        # print("Results: ", results)
        scores = sorted(results, key=lambda x: x[1], reverse=False)
        #print("Results: ", results)
        ml_scores = np.array([x[0] for x in scores])

        if pose_scores is not None:
            combined_scores = ml_scores * weight + np.array(pose_scores)[:ml_scores.shape[0]] * (1.0 - weight)
        else:
            combined_scores = ml_scores

    for i, mol_id in enumerate(mol2_code_list):
        score = combined_scores[i]
        if os.path.exists(reference):
            rmsd = rmsd_values[mol_id]
            tofile.write("{} {} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(mol2_code_list[0], i, pose_scores[i], score, ml_scores[i], rmsd))
        else:
            tofile.write("{} {} {:.3f} {:.3f} {:.3f}\n".format(mol2_code_list[0], i, pose_scores[i], score, ml_scores[i]))

    tofile.close()
'''

def extract_pose_scores(ligand_file, pose_scores_type="general"):
    if pose_scores_type == "general":
        return [0.0, ] * 1000
    elif pose_scores_type in ["idock", "ledock", 'gnina', 'tdock', 'vina', 'gnina_energy', 'gnina_cnn']:
        scores = None
        with open(ligand_file) as lines:
            if pose_scores_type == "idock":
                scores = [float(x.split()[-2]) for x in lines
                          if "REMARK 922        TOTAL FREE ENERGY PREDICTED BY IDOCK" in x]
            elif pose_scores_type == "ledock":
                scores = [float(x.split()[-2]) for x in lines if "REMARK Cluster" in x]
            elif pose_scores_type == "gnina_cnn":
                scores = [-1. * float(x.split()[-4][:-6]) for x in lines if "REMARK minimizedAffinity" in x]
                # names = [x.split()[-1].strip("\n") for x in lines if "REMARK  Name =" in x]
            elif pose_scores_type == "gnina_energy":
                scores = [-1. * float(x.split()[2][:-6]) for x in lines if "REMARK minimizedAffinity" in x]
                # names = [x.split()[-1].strip("\n") for x in lines if "REMARK  Name =" in x]
            elif pose_scores_type == "tdock":
                scores = [float(x.split()[-2]) for x in lines if "REMARK VINA ENERGY" in x]
            elif pose_scores_type.lower() == "vina":
                scores = [float(x.split()[-3]) for x in lines if "REMARK VINA RESULT:" in x]
            else:
                scores = [0.0, ] * 1000

        return scores
    else:
        return [0.0, ] * 1000

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, help="Receptor pdb/pdbqt filename.")
    parser.add_argument('-l', type=str, help="Ligand mol2 or pdbqt format filename.")
    parser.add_argument('--ref', default="refer.mol2", type=str, help="Reference Ligand mol2 filename.")
    parser.add_argument('-o', type=str, default='output.dat', help="Output filename.")
    parser.add_argument('--stype', type=str, default='vina', help="Input pose type.")
    parser.add_argument('-w', type=float, default=0.5, help="Rescore function weight. Default 0.5.")
    parser.add_argument('-m', type=str, default='sfct', help="Scoring method. Default sfct.")
    parser.add_argument('--model', type=str, default='sfct.model', help="Scoring model. Default data/sfct.model")
    parser.add_argument('--ncpus', type=int, default=1, help="Number of CPUs, default is 1.")

    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    if not os.path.exists(args.l):
        print("Docking result not found, exit now...")
        sys.exit(0)

    if os.path.exists(args.o):
        print("Output file {} exists...".format(args.o))
        sys.exit(0)

    pose_scores_original = extract_pose_scores(args.l, args.stype)

    if args.l.split(".")[-1] != "mol2":
        cnvrt_mol2 = "/tmp/ligand_{}.mol2".format(str(uuid.uuid4().hex))
        cmd = "obabel {} -O {}".format(args.l, cnvrt_mol2)
        job = sp.Popen(cmd, shell=True)
        job.communicate()
        rescoring(args.r, cnvrt_mol2, args.o, model_pkl=args.model, method=args.m,
                  reference=args.ref, pose_scores=pose_scores_original, weight=args.w,
                  ncpus=args.ncpus)
        os.remove(cnvrt_mol2)
    else:
        rescoring(args.r, args.l, args.o, model_pkl=args.model, method=args.m,
                  reference=args.ref, pose_scores=pose_scores_original, weight=args.w,
                  ncpus=args.ncpus)

