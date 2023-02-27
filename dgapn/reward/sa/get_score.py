from rdkit.Chem import RDConfig

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def get_sa_score(states):
    if not isinstance(states, list):
        return sascorer.calculateScore(states)
    else:
        return [sascorer.calculateScore(state) for state in states]
