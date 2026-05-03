import torch
import torch.nn as nn


class VisualPrototypeModule(nn.Module):
    """
    Prototype Module.
    """

    def __init__(
        self,
        num_prototypes: int = 64,
        embed_dim: int = 512,
        tau_init: float = 1.0,
        tau_min: float = 0.05,
        total_steps: int = 10 * 145,
        use_parameter_free_self_attention: bool = True,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.tau_min = tau_min
        self.tau_init = tau_init
        self.total_steps = total_steps
        self.use_parameter_free_self_attention = use_parameter_free_self_attention

        # Learnable visual meta matrix Q in R^{N x D}
        self.visual_meta_matrix = nn.Parameter(torch.empty(num_prototypes, embed_dim))
        nn.init.uniform_(self.visual_meta_matrix, -0.1, 0.1)

        self.register_buffer("tau", torch.tensor(tau_init))

    def update_tau(self, current_step: int):
        """Linearly decay tau from tau_init to tau_min over total_steps."""
        progress = min(max(current_step, 0), self.total_steps) / self.total_steps
        new_tau = self.tau_init + progress * (self.tau_min - self.tau_init)
        self.tau.fill_(max(new_tau, self.tau_min))

    def _compute_query_group(self) -> torch.Tensor:
        """
        Q* = (Q Q^T) Q when parameter-free self-attention is enabled.
        """
        q_matrix = self.visual_meta_matrix
        if not self.use_parameter_free_self_attention:
            return q_matrix
        qqt = torch.mm(q_matrix, q_matrix.t())
        q_star = torch.mm(qqt, q_matrix)
        return q_star

    def forward(
        self,
        visual_context: torch.Tensor,
        training: bool = True,
        current_step: int = None,
    ) -> torch.Tensor:
        """
        Produce a prototype query vector for each sample in the batch.
        """
        if current_step is not None:
            self.update_tau(current_step)

        q_star = self._compute_query_group()
        # Keep all prototype ops on a single dtype/device (fp16 path included).
        q_star = q_star.to(dtype=visual_context.dtype, device=visual_context.device)
        tau = self.tau.to(dtype=visual_context.dtype, device=visual_context.device)
        scores = torch.matmul(visual_context, q_star.t()) / tau
        weights = torch.softmax(scores, dim=-1)

        if training:
            return torch.matmul(weights, q_star)

        indices = weights.argmax(dim=-1)
        return q_star[indices]
