import torch
import random
import numpy as np
from network import SuperTrack

def set_random_seed(seed):
    print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_random_seed(3407)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SuperTrack(device)

    model.train()


if __name__ == "__main__":
    main()