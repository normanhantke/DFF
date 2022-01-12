import torch

if not torch.cuda.is_available():
    sys.exit(77);
