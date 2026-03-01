import random
import numpy as np
import torch


def worker_init_fn(wid):
    """
    Initializes seeds for Python random, NumPy, and PyTorch for DataLoader workers.
    """
    # Generate a unique seed sequence per worker
    seed_sequence = np.random.SeedSequence([torch.initial_seed(), wid])

    # Generate independent seeds for each random generator
    torch_seed = seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint32)[0]
    np_seed = seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint32)[0]
    py_seed = seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint32)[0]

    # Apply seeds
    torch.manual_seed(int(torch_seed))
    np.random.seed(int(np_seed))
    random.seed(int(py_seed))
