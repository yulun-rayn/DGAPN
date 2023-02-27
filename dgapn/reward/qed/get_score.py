import rdkit.Chem.QED as QED

def get_qed_score(states):
    if not isinstance(states, list):
        return QED.qed(states)
    else:
        return [QED.qed(state) for state in states]
