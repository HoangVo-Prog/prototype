import torch
import torch.nn as nn


class VisualPrototypeModule(nn.Module):
    """
    Prototype Module
    """

    def __init__(self, num_prototypes: int = 64, embed_dim: int = 512,
                 tau_init: float = 1.0, tau_min: float = 0.05,
                 total_steps: int = 10 * 145):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.tau_min = tau_min
        self.tau_init = tau_init
        self.total_steps = total_steps

        # Learnable visual meta matrix  Q ∈ R^{N x D}
        self.visual_meta_matrix = nn.Parameter(
            torch.empty(num_prototypes, embed_dim)
        )
        nn.init.uniform_(self.visual_meta_matrix, -0.1, 0.1)

        self.register_buffer('tau', torch.tensor(tau_init))

    # ------------------------------------------------------------------
    # Temperature schedule (call once per training step)
    # ------------------------------------------------------------------
    def update_tau(self, current_step: int):
        """Linearly decay tau from tau_init to tau_min over total_steps."""
        progress = min(max(current_step, 0), self.total_steps) / self.total_steps
        new_tau = self.tau_init + progress * (self.tau_min - self.tau_init)
        self.tau.fill_(max(new_tau, self.tau_min))

    # ------------------------------------------------------------------
    # Core: parameter-free self-attention on Q 
    # ------------------------------------------------------------------
    def _compute_query_group(self) -> torch.Tensor:
        """
        Q* = (Q Q^T) Q    — parameter-free self-attention.
        Returns Q* ∈ R^{N x D}.
        """
        Q = self.visual_meta_matrix                   # [N, D]
        QQT = torch.mm(Q, Q.t())                      # [N, N]
        Q_star = torch.mm(QQT, Q)                     # [N, D]
        return Q_star

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self,
                visual_context: torch.Tensor,
                training: bool = True,
                current_step: int = None) -> torch.Tensor:
        """
        Produce a prototype query vector for each sample in the batch.
        """
        if current_step is not None:
            self.update_tau(current_step)

        Q_star = self._compute_query_group()          # [N, D]

        # k_i = softmax( e_c · q_i^T / tau )
        scores = torch.matmul(visual_context, Q_star.t()) / self.tau  # [B, N]
        weights = torch.softmax(scores, dim=-1)                        # [B, N]

        if training:
            # Soft query: q̃* = k Q* 
            soft_query = torch.matmul(weights, Q_star)                 # [B, D]
            return soft_query
        else:
            # Hard query: select the row with maximum weight 
            indices = weights.argmax(dim=-1)                           # [B]
            hard_query = Q_star[indices]                               # [B, D]
            return hard_query
        