import os
import gym
import random
from rdkit import Chem
from crem.crem import mutate_mol, grow_mol

from dataset.get_dataset import download_dataset, unpack_dataset
from dataset import preprocess
from utils.graph_utils import mol_to_pyg_graph

DATASET_URL = "http://www.qsar4u.com/files/cremdb/replacements02_sc2.db.gz"
DATASET_NAME = 'replacements02_sc2.db'

class CReM_Env(object):
    def __init__(self,
                 data_path,
                 warm_start_dataset = None,
                 nb_sample_crem=16,
                 nb_cores=1,
                 max_timesteps=12,
                 mode='mol'):

        self.grow = True if warm_start_dataset is None else False
        if not self.grow:
            self.scores, self.smiles = preprocess.main(os.path.join(data_path, warm_start_dataset))

        self.nb_sample_crem = nb_sample_crem
        self.nb_cores = nb_cores
        self.max_timesteps = max_timesteps
        self.mode = mode

        _ = download_dataset(data_path,
                             DATASET_NAME+'.gz',
                             DATASET_URL)
        self.db_fname = unpack_dataset(data_path,
                                       DATASET_NAME+'.gz',
                                       DATASET_NAME)

        self.timestep = 0

    def reset(self, mol=None, include_current_state=True, reset_timestep=True, return_type=None):
        if reset_timestep:
            self.timestep = 0

        if mol is None:
            if self.grow:
                carbon = Chem.MolFromSmiles("C")
                smiles = list(grow_mol(carbon, 
                                        db_name=self.db_fname,
                                        ncores=self.nb_cores))
                mol = Chem.MolFromSmiles(random.choice(smiles))
            else:
                idx = random.randrange(len(self.scores))
                mol = Chem.MolFromSmiles(self.smiles[idx])
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        if return_type is None:
            return_type = self.mode
        return self.mol_to_candidates(mol, include_current_state, return_type)

    def step(self, action, include_current_state=True, return_type=None):
        self.timestep += 1

        mol = self.new_mols[action]

        if return_type is None:
            return_type = self.mode
        return self.mol_to_candidates(mol, include_current_state, return_type)

    def mol_to_candidates(self, mol, include_current_state, return_type):
        mol_candidates, done = self.get_crem_candidates(mol, include_current_state)
        if return_type == 'pyg':
            pyg = mol_to_pyg_graph(mol)[0]
            pyg_candidates = [mol_to_pyg_graph(i)[0] for i in mol_candidates]
            return pyg, pyg_candidates, done
        if return_type == 'smiles':
            smiles = Chem.MolToSmiles(mol)
            smiles_candidates = [Chem.MolToSmiles(mol) for mol in mol_candidates]
            return smiles, smiles_candidates, done

        return mol, mol_candidates, done

    def get_crem_candidates(self, mol, include_current_state):

        try:
            new_mols = list(mutate_mol(mol,
                                       self.db_fname,
                                       max_replacements = self.nb_sample_crem,
                                       return_mol=True,
                                       ncores=self.nb_cores))
            # print("CReM options:" + str(len(new_mols)))
            new_mols = [Chem.RemoveHs(i[1]) for i in new_mols]
        except Exception as e:
            print("CReM forward error: " + str(e))
            print("SMILE: " + Chem.MolToSmiles(mol))
            new_mols = []
        self.new_mols = [mol] + new_mols if include_current_state else new_mols
        mol_candidates = self.new_mols

        if len(mol_candidates)==0: #if len(mol_candidates)==include_current_state:
            return None, True
        elif self.timestep >= self.max_timesteps:
            return mol_candidates, True
        else:
            return mol_candidates, False
