from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp

if __name__ == "__main__":
    START_EXP_IDX = 1
    NUM_EXP = 3
    NUM_POISONED_WORKERS = 20  # M=20
    REPLACEMENT_METHOD = replace_1_with_9  # Source class = 1
    KWARGS = {"NUM_WORKERS_PER_ROUND": 5}

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(
            REPLACEMENT_METHOD,
            NUM_POISONED_WORKERS,
            KWARGS,
            RandomSelectionStrategy(),
            experiment_id,
        )
