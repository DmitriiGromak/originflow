import torch
def set_global_seed(seed=42):
    """
    Set the seed for all random number generators for reproducibility.
    
    Args:
        seed (int): The seed to use
    """
    import random
    import numpy as np
    import torch
    
    # Set Python's random seed
    random.seed(seed)
    Q
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # Set CUDA's random seed if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # Additional settings for CUDA determinism
        # Note: This may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Global random seed set to {seed}")

# Replace the incorrect torch.seed(123) with:
set_global_seed(123)