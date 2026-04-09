"""
kd_lora_tree_ortho.py
─────────────────────
Drop-in replacement for kd_lora_tree.py that:
  1. Keeps 100 % of the original Tree_LoRA logic.
  2. Adds a `get_ortho_basis` helper so Tree_LoRA_Ortho can build proper
     multi-rank orthogonal projectors (SVD-based, not just rank-1).
  3. Exposes `gradient_stats()` for visualisation – returns cosine-sim and
     orthogonal-residual-norm between current grad and all previous tasks,
     per LoRA layer.

Nothing in the original KD_LoRA_Tree is removed; only additive changes.
"""

import copy
import math

import torch

from utils.utils import print_rank_0


# ══════════════════════════════════════════════════════════════════════════════
# KD-Tree node  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

class KDTreeNode:
    def __init__(self, task_indices, depth, grads_tensor, lora_depth):
        self.task_indices    = task_indices
        self.depth           = depth
        self.left            = None
        self.right           = None
        self.is_leaf         = False
        self.lora_depth      = lora_depth
        self.mean_vector     = None
        self.median_similarity = None
        self.build_node(grads_tensor)

    def build_node(self, grads_tensor):
        if self.depth >= self.lora_depth or len(self.task_indices) <= 1:
            self.is_leaf = True
            return

        current_grads = grads_tensor[self.task_indices, self.depth, :]
        self.mean_vector = current_grads.mean(dim=0)
        similarities     = torch.mv(current_grads, self.mean_vector)
        self.median_similarity = torch.median(similarities).item()

        left_indices  = [self.task_indices[i] for i in range(len(self.task_indices))
                         if similarities[i].item() >= self.median_similarity]
        right_indices = [self.task_indices[i] for i in range(len(self.task_indices))
                         if similarities[i].item() < self.median_similarity]

        if len(left_indices) == 0 or len(right_indices) == 0:
            mid           = len(self.task_indices) // 2
            left_indices  = self.task_indices[:mid]
            right_indices = self.task_indices[mid:]

        self.left  = KDTreeNode(left_indices,  self.depth + 1, grads_tensor, self.lora_depth)
        self.right = KDTreeNode(right_indices, self.depth + 1, grads_tensor, self.lora_depth)

    def __str__(self, level=0):
        indent = "  " * level
        if self.is_leaf:
            return f"{indent}Leaf(depth={self.depth}, tasks={self.task_indices})\n"
        mean_list = self.mean_vector[:2].tolist()
        mean_str  = ", ".join([f"{x:.4f}" for x in mean_list])
        result = (
            f"{indent}Node(depth={self.depth}, tasks={self.task_indices}, "
            f"mean_vector=[{mean_str}, ...], "
            f"median_similarity={self.median_similarity:.4f})\n"
        )
        if self.left:
            result += self.left.__str__(level + 1)
        if self.right:
            result += self.right.__str__(level + 1)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Loss helpers  (unchanged from original + new ortho variant)
# ══════════════════════════════════════════════════════════════════════════════

def tree_lora_loss(current_grad, all_grad, task_id, prev_id_matrix,
                   multiple_module=True):
    """Original Tree_LoRA similarity loss (unchanged)."""
    reg_loss = None
    if multiple_module:
        for depth_id, prev_task_id in enumerate(prev_id_matrix):
            term = -(current_grad[depth_id] * all_grad[prev_task_id][depth_id]).sum()
            reg_loss = term if reg_loss is None else reg_loss + term
    else:
        prev_id  = prev_id_matrix[0]
        reg_loss = -(current_grad.reshape(-1) * all_grad[prev_id].reshape(-1)).sum()
    return reg_loss


def ortho_projection_loss(current_grad, all_accumulate_grads, task_id,
                           lambda_orth=0.1, device='cuda', eps=1e-8):
    """
    NEW: Orthogonal Projection Loss.

    For every previous task k and every LoRA depth d we compute the
    squared projection of g_t_d onto v_k_d:

        cost_{k,d} = dot(g_t_d, v_k_d)^2 / ||v_k_d||^2

    Summed over k and d, scaled by lambda_orth / (K * D).

    Parameters
    ----------
    current_grad        : (lora_depth, D) – differentiable parameter tensor
    all_accumulate_grads: list of (lora_depth, D) or None, length >= task_id
    task_id             : int
    lambda_orth         : float
    device              : str or torch.device
    eps                 : float  numerical safety

    Returns
    -------
    orth_loss            : scalar differentiable Tensor
    per_layer_proj_norms : list[list[float]]  [prev_k][depth_d]
    """
    lora_depth = current_grad.shape[0]
    orth_loss  = torch.tensor(0.0, device=device)
    per_layer_proj_norms = []

    prev_grads = all_accumulate_grads[:task_id]
    K = sum(1 for g in prev_grads if g is not None)
    if K == 0:
        return orth_loss, per_layer_proj_norms

    for v_k in prev_grads:
        if v_k is None:
            continue
        v_k_dev  = v_k.to(device, non_blocking=True)
        norms_k  = []
        for d in range(lora_depth):
            g_d       = current_grad[d]
            v_d       = v_k_dev[d].detach()
            v_norm_sq = torch.dot(v_d, v_d) + eps
            dot_val   = torch.dot(g_d, v_d)          # differentiable
            proj_sq   = dot_val ** 2 / v_norm_sq
            orth_loss = orth_loss + proj_sq
            norms_k.append(proj_sq.item() ** 0.5)
        per_layer_proj_norms.append(norms_k)

    orth_loss = lambda_orth * orth_loss / (K * lora_depth)
    return orth_loss, per_layer_proj_norms


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class KD_LoRA_Tree_Ortho:
    """
    Extends KD_LoRA_Tree with:
      • `get_ortho_basis(task_id, device)`  – SVD-based multi-rank projector
      • `gradient_stats(g_current, task_id, device)` – cosine-sim + orth-norms
      • `ortho_loss(g_current, task_id, lambda_orth, device)` – convenience wrapper

    Everything else is identical to the original KD_LoRA_Tree.
    """

    def __init__(self, args):
        self.root                = None
        self.mask                = None
        self.mask_tensor         = None
        self.last_task_id        = -1
        self.args                = args
        self.all_grad_device     = None
        self.all_accumulate_grads = [None] * self.args.num_tasks
        self.num_of_selected     = None
        self.kd_tree_root        = None
        self.current_grad        = None
        self.sim                 = None

    # ------------------------------------------------------------------
    # identical to original
    # ------------------------------------------------------------------

    def new_epoch_init(self, train_dataloader_len):
        self.current_grad  = None
        self.all_grad      = None
        self.num_of_selected = None
        self.tmp_rounds    = -1
        self.total_rounds  = train_dataloader_len
        self.sim           = None

    def end_task(self, task_id):
        if self.args.reg > 0:
            self.all_accumulate_grads[task_id] = self.current_grad

        lora_depth  = self.current_grad.shape[0]
        print_rank_0(f"\nUpdating the KD Tree with task {task_id}...",
                     self.args.global_rank)

        valid_grads = [g for g in self.all_accumulate_grads[:task_id + 1]
                       if g is not None]
        if not valid_grads:
            print("No gradients to build the tree.")
            return

        grads_tensor = copy.deepcopy(torch.stack(valid_grads))
        for i in range(grads_tensor.shape[0] - 1, 0, -1):
            grads_tensor[i] = grads_tensor[i] - grads_tensor[i - 1]

        task_ids = [i for i, g in enumerate(self.all_accumulate_grads[:task_id + 1])
                    if g is not None]

        self.kd_tree_root = KDTreeNode(
            task_indices=task_ids,
            depth=0,
            grads_tensor=grads_tensor,
            lora_depth=lora_depth
        )
        print_rank_0("KD Tree updated successfully.", self.args.global_rank)
        print_rank_0(self.kd_tree_root, self.args.global_rank)

    def step(self):
        self.tmp_rounds += 1
        self.tmp_reg = self.args.reg * self.tmp_rounds / self.total_rounds

    def insert_grad(self, _grad_current):
        frac = 1.0 / self.total_rounds
        for _ in range(len(_grad_current)):
            if self.current_grad is None:
                self.current_grad = _grad_current.detach() * frac
            else:
                self.current_grad = self.current_grad + _grad_current.detach() * frac

    def get_mask(self, class_mask, task_id, args, logits):
        if self.mask is None or task_id != self.last_task_id:
            self.last_task_id = task_id
            self.mask         = class_mask[task_id]
            self.mask_tensor  = torch.full(
                (args.nb_classes,), False, dtype=torch.bool, device=logits.device
            )
            self.mask_tensor[self.mask] = True

    def tree_search(self, task_id, device):
        cosine_sim = False

        if self.all_grad is None:
            self.all_grad = torch.stack(
                self.all_accumulate_grads[:task_id], dim=0
            ).to(device, non_blocking=True)
            self.all_grad_device = self.all_grad

            if cosine_sim:
                self.all_grad = self.all_grad / (
                    torch.norm(self.all_grad, dim=2).mean(dim=0) + 1e-5
                ).unsqueeze(0).unsqueeze(2)

            if self.sim is None:
                self.sim = torch.zeros(
                    (task_id, self.all_grad.shape[1]), device=device
                )
                self.num_of_selected = torch.zeros(
                    self.args.num_tasks, self.all_grad.shape[1]
                ).to(device, non_blocking=True)

        sim = self.sim.clone()
        valid_mask = self.num_of_selected[:task_id, :] > 0
        sim[valid_mask] = sim[valid_mask] / self.num_of_selected[:task_id, :][valid_mask]

        explore = (
            1.0 / torch.sqrt(2 * self.num_of_selected[:task_id, :] + 1e-5)
            * math.sqrt(math.log(
                2 * self.total_rounds
                * (self.tmp_rounds + 1)
                * (self.tmp_rounds + 2)
            ))
        )

        if cosine_sim:
            sim += explore
        else:
            sim -= explore
            sim  = -sim

        sim += torch.min(sim)

        first_idx = torch.multinomial(
            torch.softmax(torch.sum(sim, dim=1), dim=0),
            num_samples=1, replacement=True
        ).item()

        similarity = 1.0
        if self.kd_tree_root is not None and self.kd_tree_root.left is not None:
            if first_idx in self.kd_tree_root.left.task_indices:
                similarity = (self.kd_tree_root.left.median_similarity
                              if self.kd_tree_root.left.median_similarity is not None
                              else 1.0)
                sim[self.kd_tree_root.left.task_indices] *= min(similarity, 1.5)
            else:
                similarity = (self.kd_tree_root.right.median_similarity
                              if self.kd_tree_root.right.median_similarity is not None
                              else 1.0)
                sim[self.kd_tree_root.right.task_indices] *= min(similarity, 1.5)

        if self.tmp_rounds % 100 == 0:
            print_rank_0(
                f'\033[34m****first idx: {first_idx}, similarity: {similarity}\033[0m',
                self.args.global_rank
            )

        sim = sim / (torch.max(sim) - torch.min(sim) + 1e-5)
        sim[task_id:, :] = -torch.inf
        sim_normalized   = torch.softmax(sim, dim=0)

        prev_id_matrix = torch.multinomial(
            sim_normalized.T, num_samples=1, replacement=True
        ).reshape(-1)

        self.num_of_selected[prev_id_matrix, torch.arange(sim.shape[1])] += 1
        self.update_similarity(prev_id_matrix, device)
        return prev_id_matrix

    def get_loss(self, _grad_current, loss, task_id, prev_id_matrix):
        reg_loss = tree_lora_loss(
            _grad_current, self.all_grad_device, task_id, prev_id_matrix
        )
        reg_loss = (
            reg_loss / (reg_loss.detach().clone() + 1e-5)
            * loss.detach().clone()
            * self.tmp_reg
        )
        return reg_loss

    def update_similarity(self, prev_id_matrix, device):
        if self.sim is None:
            return
        for depth_idx, prev_id in enumerate(prev_id_matrix):
            self.sim[prev_id, depth_idx] -= torch.sum(
                torch.abs(
                    self.current_grad[depth_idx]
                    - self.all_grad[prev_id, depth_idx]
                )
            ).item()

    # ------------------------------------------------------------------
    # NEW: orthogonal projector helpers
    # ------------------------------------------------------------------

    def get_ortho_basis(self, task_id, device, eps=1e-8):
        """
        Build an SVD-based orthonormal basis for the subspace spanned by
        the accumulated gradients of tasks 0 … task_id-1, per LoRA layer.

        Returns
        -------
        bases : list of Tensor  length = lora_depth
                Each element is  (D, r_k)  where r_k <= task_id is the
                number of linearly-independent directions found.

        Usage (projection onto subspace):
            B = bases[d]              # (D, r)
            proj = B @ (B.T @ g_d)   # (D,)  component inside subspace
            orth = g_d - proj         # (D,)  orthogonal residual
        """
        prev_grads = [g for g in self.all_accumulate_grads[:task_id] if g is not None]
        if not prev_grads:
            return None

        lora_depth = prev_grads[0].shape[0]
        bases = []

        for d in range(lora_depth):
            # Stack all previous grad vectors for layer d → (K, D)
            V = torch.stack([g[d] for g in prev_grads], dim=0).to(device)
            # SVD: V = U S Wt  →  columns of Wt form a basis
            try:
                _, S, Vh = torch.linalg.svd(V, full_matrices=False)
            except Exception:
                # Fallback: just normalise each row
                Vh = V / (torch.norm(V, dim=1, keepdim=True) + eps)
                S  = torch.ones(Vh.shape[0], device=device)

            # Keep only directions with significant singular values
            threshold = eps * S[0]
            mask      = S > threshold
            basis_d   = Vh[mask].T   # (D, r)
            bases.append(basis_d)

        return bases  # list of (D, r_k) tensors

    def ortho_loss(self, g_current, task_id, lambda_orth=0.1, device='cuda'):
        """
        Convenience wrapper around the module-level ortho_projection_loss.
        Calls the rank-1 projector version (matches Tree_LoRA_Ortho default).
        """
        return ortho_projection_loss(
            g_current, self.all_accumulate_grads,
            task_id, lambda_orth=lambda_orth, device=device
        )

    # ------------------------------------------------------------------
    # NEW: gradient statistics for visualisation
    # ------------------------------------------------------------------

    def gradient_stats(self, g_current, task_id, device, eps=1e-8):
        """
        Compute per-layer, per-previous-task gradient statistics.

        Returns
        -------
        stats : list of dicts, one per previous task k
            {
              'task_k'       : int,
              'cos_sim'      : list[float]   length = lora_depth,
              'proj_norm'    : list[float]   length = lora_depth,
              'orth_norm'    : list[float]   length = lora_depth,
              'grad_norm_cur': list[float]   length = lora_depth,
              'grad_norm_ref': list[float]   length = lora_depth,
            }

        cos_sim[d]   : cosine similarity between g_t_d and g_k_d
        proj_norm[d] : ||P_k_d g_t_d||  – norm of component IN subspace of task k
        orth_norm[d] : ||g_t_d - P_k_d g_t_d||  – norm of ORTHOGONAL residual
        """
        lora_depth = g_current.shape[0]
        stats      = []

        for k, v_k in enumerate(self.all_accumulate_grads[:task_id]):
            if v_k is None:
                continue
            v_k_dev = v_k.to(device, non_blocking=True).detach()
            entry   = {'task_k': k, 'cos_sim': [], 'proj_norm': [],
                       'orth_norm': [], 'grad_norm_cur': [], 'grad_norm_ref': []}

            for d in range(lora_depth):
                g_d = g_current[d].detach()
                v_d = v_k_dev[d]

                g_norm = torch.norm(g_d).item()
                v_norm = torch.norm(v_d).item()

                cos_val  = (torch.dot(g_d, v_d) / (g_norm * v_norm + eps)).item()

                # rank-1 projection
                v_norm_sq  = v_norm ** 2 + eps
                proj_vec   = (torch.dot(g_d, v_d) / v_norm_sq) * v_d
                proj_n     = torch.norm(proj_vec).item()
                orth_n     = torch.norm(g_d - proj_vec).item()

                entry['cos_sim'].append(cos_val)
                entry['proj_norm'].append(proj_n)
                entry['orth_norm'].append(orth_n)
                entry['grad_norm_cur'].append(g_norm)
                entry['grad_norm_ref'].append(v_norm)

            stats.append(entry)

        return stats