from .logp.get_score import get_logp_score, get_penalized_logp
from .qed.get_score import get_qed_score
from .sa.get_score import get_sa_score
from .adtgpu.get_score import get_dock_score

def get_reward(states, reward_type, args=None):

    if reward_type == 'logp':
        return get_logp_score(states)

    elif reward_type == 'plogp':
        return get_penalized_logp(states)

    elif reward_type == 'qed':
        qed = get_qed_score(states)
        # scale QED to 0-10
        if isinstance(qed, list):
            return [10.* (s-0.009)/0.939 for s in qed]
        else:
            return 10.* (qed-0.009)/0.939

    elif reward_type == 'sa':
        sa = get_sa_score(states)
        # scale SA to 0-10
        if isinstance(sa, list):
            return [10.* (10.-s)/9. for s in sa]
        else:
            return 10.* (10.-sa)/9.

    elif reward_type == 'dock':
        dock = get_dock_score(states, args=args)
        # negative dock
        if isinstance(dock, list):
            return [-s for s in dock]
        else:
            return -dock

    else:
        raise ValueError("Reward type not recognized.")
