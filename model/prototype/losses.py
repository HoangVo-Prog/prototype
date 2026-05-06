import torch
import torch.nn.functional as F


def compute_diversity_loss(prototype_bank: torch.Tensor) -> torch.Tensor:
    """
    Penalize prototype collapse by minimizing off-diagonal cosine similarity.
    """
    prototype_bank = F.normalize(prototype_bank, p=2, dim=-1)
    similarity = torch.matmul(prototype_bank, prototype_bank.t())
    num_prototypes = similarity.size(0)
    if num_prototypes <= 1:
        return similarity.new_zeros(())

    off_diagonal_mask = ~torch.eye(num_prototypes, dtype=torch.bool, device=similarity.device)
    off_diagonal_similarity = similarity[off_diagonal_mask]
    return off_diagonal_similarity.pow(2).mean()
