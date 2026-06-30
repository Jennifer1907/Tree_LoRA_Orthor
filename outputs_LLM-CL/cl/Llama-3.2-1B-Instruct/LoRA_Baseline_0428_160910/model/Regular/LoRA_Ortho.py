import json
import os

import torch
from tqdm import tqdm

from model.base_model import CL_Base_Model
from utils.kd_lora_tree import KD_LoRA_Tree
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device


# ---------------------------------------------------------------------------
# Helpers  (identical to Tree_LoRA_Ortho; copied for self-containment)
# ---------------------------------------------------------------------------

def _collect_loranew_A_params(model):
    """Return (name, param) pairs for every loranew_A parameter, stable order."""
    return [(n, p) for n, p in model.named_parameters() if "loranew_A" in n]


def _stack_grad_via_autograd(loss_ce, params_and_names):
    """
    Compute autograd.grad of loss_ce w.r.t. loranew_A params.
    retain_graph=True so model.backward() can still run afterwards.
    Returns (D, flat_dim) float32 tensor (no gradient tracking).
    """
    param_list = [p for _, p in params_and_names]
    grads = torch.autograd.grad(
        loss_ce,
        param_list,
        retain_graph=True,   # CRITICAL: must retain for subsequent backward()
        create_graph=False,
        allow_unused=True,
    )
    g_list = []
    for g, (_, p) in zip(grads, params_and_names):
        if g is None:
            g_list.append(torch.zeros(p.numel(), dtype=torch.float32))
        else:
            g_list.append(g.float().reshape(-1))
    return torch.stack(g_list, dim=0)   # (D, flat_dim)


def _collect_real_grads(params_and_names):
    """Gather .grad buffers after backward(). Returns (D, flat_dim) or None."""
    grads = []
    for _, p in params_and_names:
        if p.grad is None:
            return None
        grads.append(p.grad.detach().float().reshape(-1))
    return torch.stack(grads, dim=0)


# ---------------------------------------------------------------------------
# Orthogonal Projection Loss  (stop-gradient / GradOrth trick)
# Identical implementation to Tree_LoRA_Ortho._orth_loss_stop_grad
# ---------------------------------------------------------------------------

def _orth_loss_stop_grad(params_and_names, g_t_detached, all_prev_grads, reg_orth):
    """
    L_orth = (reg_orth / (K*D)) * Σ_k Σ_d  α_{k,d} * <param_d_flat, v_{k,d}>

    where  α_{k,d} = <g_t_d (detached), v_{k,d}> / ||v_{k,d}||²   (scalar, detached)

    Args:
        params_and_names : list[(name, param)] for loranew_A layers
        g_t_detached     : (D, flat_dim) float32 CPU, DETACHED
        all_prev_grads   : (K, D, flat_dim) float32 CPU
        reg_orth         : float

    Returns:
        orth_loss  : differentiable scalar (through params)
        proj_norms : list[float] of length D
    """
    K, D, _ = all_prev_grads.shape
    orth_loss  = None
    proj_norms = []

    for d, (_, param) in enumerate(params_and_names):
        param_flat = param.reshape(-1).float()   # differentiable, on CUDA
        g_d        = g_t_detached[d]             # CPU, detached

        layer_max_proj = 0.0
        for k in range(K):
            v_k_d    = all_prev_grads[k, d]
            v_norm_sq = torch.dot(v_k_d, v_k_d).item()
            if v_norm_sq < 1e-12:
                continue

            alpha = (torch.dot(g_d, v_k_d) / v_norm_sq).detach()  # scalar CPU

            v_k_d_cuda = v_k_d.to(param_flat.device)
            term = alpha.to(param_flat.device) * torch.dot(param_flat, v_k_d_cuda)

            orth_loss = term if orth_loss is None else orth_loss + term

            proj_norm = (alpha.item() ** 2 * v_norm_sq) ** 0.5
            layer_max_proj = max(layer_max_proj, proj_norm)

        proj_norms.append(layer_max_proj)

    if orth_loss is None:
        return torch.tensor(0.0), proj_norms

    scale = reg_orth / max(K * D, 1)
    return orth_loss * scale, proj_norms


# ---------------------------------------------------------------------------
# Ortho_LoRA
# ---------------------------------------------------------------------------
# Ablation variant: LoRA + Orthogonal Projection Loss ONLY.
# The KD-tree gradient-similarity regulariser (reg_loss) is removed.
#
# Loss at every step:
#     total_loss = loss_ce  +  orth_loss
#
# Purpose: isolate the contribution of OPL from the KD-tree component.
# Answers: "does OPL alone reduce forgetting?"
#
# Kept vs Tree_LoRA_Ortho:
#   - autograd.grad  call   (needed to compute g_t for OPL)
#   - gradient accumulation (needed to store prev-task grad vectors for OPL)
#   - _orth_loss_stop_grad  (unchanged)
#   - vis_log               (same schema; orth_loss + proj_norms)
#
# Removed vs Tree_LoRA_Ortho:
#   - KD_LoRA_Tree.tree_search / get_loss  → reg_loss term
#   - KD_LoRA_Tree.step / new_epoch_init / end_task  → tree not needed
#   - basis_usage_per_layer visualisation field (tree-specific)
# ---------------------------------------------------------------------------

class Ortho_LoRA(CL_Base_Model):
    """
    LoRA + Orthogonal Projection Loss (no KD-tree similarity regulariser).

    total_loss = loss_ce  +  orth_loss

    Extra args (with defaults):
        args.reg_orth     - weight for OPL           (default 0.1)
        args.vis_interval - steps between vis entries (default 50)

    Note: args.reg is ignored / forced to 0 so shared arg-parsing is unaffected.
    """

    def __init__(self, model, tokenizer, optimizer,
                 train_task_list, eval_task_list, test_task_list, args,
                 lamda_1=0.5, lamda_2=0):
        super().__init__(model, tokenizer, optimizer,
                         train_task_list, eval_task_list, test_task_list, args)

        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.tiktok  = TIKTOK(args)

        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

        self.reg_orth     = getattr(args, "reg_orth",     0.1)
        self.vis_interval = getattr(args, "vis_interval", 50)

        # KD-tree is used ONLY as a gradient store (all_accumulate_grads).
        # tree_search / get_loss are never called.
        num_task       = len(self.train_task_list)
        args.num_tasks = num_task
        # Reuse KD_LoRA_Tree purely for its gradient accumulation bookkeeping
        # (insert_grad, end_task, all_accumulate_grads).
        self._grad_store = KD_LoRA_Tree(args)

        # Force-disable KD-tree similarity reg (keeps arg-parsing clean)
        self.args.reg = 0

        self._vis_log     = []
        self._accum_grad  = None
        self._accum_steps = 0

    # ------------------------------------------------------------------
    # Gradient accumulation  (identical to Tree_LoRA_Ortho)
    # ------------------------------------------------------------------

    def _reset_grad_accum(self):
        self._accum_grad  = None
        self._accum_steps = 0

    def _accumulate_real_grad(self, params_and_names):
        g = _collect_real_grads(params_and_names)
        if g is None:
            return
        g_cpu = g.cpu()
        if self._accum_grad is None:
            self._accum_grad = g_cpu.clone()
        else:
            self._accum_grad.add_(g_cpu)
        self._accum_steps += 1

    def _finalize_task_grad(self, task_id):
        """Store per-task mean gradient into the grad store."""
        if self._accum_steps == 0 or self._accum_grad is None:
            print_rank_0(
                f"[WARNING] No gradients accumulated for task {task_id}.",
                self.args.global_rank)
            return
        mean_grad = self._accum_grad / self._accum_steps
        self._grad_store.current_grad = mean_grad.to(self.device)

    # ------------------------------------------------------------------
    # Previous-task gradient tensor
    # ------------------------------------------------------------------

    def _get_all_prev_grads_cpu(self, task_id):
        """
        Stack accumulated gradients for tasks 0 … task_id-1.
        Returns (K, D, flat_dim) float32 CPU tensor, or None.
        """
        prev = [g for g in self._grad_store.all_accumulate_grads[:task_id]
                if g is not None]
        if not prev:
            return None
        return torch.stack(prev, dim=0).float().cpu()  # (K, D, flat_dim)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_task(self, task, task_id, epochs):
        train_dataloader     = self.train_task_list[task]
        train_dataloader_len = len(train_dataloader)
        total_steps          = epochs * train_dataloader_len
        progress_bar = tqdm(total=total_steps, leave=True,
                            disable=(self.args.global_rank != 0))

        self._vis_log = []

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, "
                f"Total Micro Batches {train_dataloader_len}",
                self.args.global_rank)

            self.model.train()
            self.tiktok.print_time(self.args.global_rank)

            # Initialise grad store's epoch counters (needed by insert_grad)
            self._grad_store.new_epoch_init(train_dataloader_len)
            self._reset_grad_accum()

            # Stable param order within epoch
            params_and_names = _collect_loranew_A_params(self.model)

            tmp_rounds = -1

            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1

                del batch['sources']
                batch = to_device(batch, self.device)

                # ── Step 1: Forward ───────────────────────────────────────
                outputs = self.model(**batch, use_cache=False)
                loss_ce = outputs.loss

                orth_loss          = torch.tensor(0.0, device=self.device)
                proj_norms         = []
                all_prev_grads_cpu = None

                if task_id > 0 and self.reg_orth > 0:
                    # ── Step 2: Compute g_t via autograd ──────────────────
                    #    retain_graph=True so backward() still works later
                    self.tiktok.tik()
                    g_t = _stack_grad_via_autograd(loss_ce, params_and_names)
                    self.tiktok.tok("autograd_grad @T{} E{}".format(task_id, epoch))

                    # Also feed g_t to the grad store (for end-of-task storage)
                    self._grad_store.insert_grad(g_t.to(self.device))

                    # ── Step 3: OPL ───────────────────────────────────────
                    all_prev_grads_cpu = self._get_all_prev_grads_cpu(task_id)

                    if all_prev_grads_cpu is not None:
                        self.tiktok.tik()
                        g_t_det_cpu = g_t.detach().cpu()

                        orth_loss, proj_norms = _orth_loss_stop_grad(
                            params_and_names,
                            g_t_det_cpu,
                            all_prev_grads_cpu,
                            self.reg_orth,
                        )
                        orth_loss = orth_loss.to(self.device)
                        self.tiktok.tok("orth_loss @T{} E{}".format(task_id, epoch))
                    else:
                        proj_norms = [0.0] * len(params_and_names)

                    # Periodic console log
                    if tmp_rounds % 100 == 0:
                        print_rank_0(
                            "\033[35m[OrthoLoRA] orth={:.4f}\033[0m".format(
                                orth_loss.item()),
                            self.args.global_rank)

                elif task_id == 0:
                    # First task: no previous grads → still insert for storage
                    self.tiktok.tik()
                    g_t = _stack_grad_via_autograd(loss_ce, params_and_names)
                    self.tiktok.tok("autograd_grad @T{} E{}".format(task_id, epoch))
                    self._grad_store.insert_grad(g_t.to(self.device))

                # ── Step 4: Combine losses ────────────────────────────────
                # NOTE: NO reg_loss term here — this is the key ablation diff
                total_loss = loss_ce + orth_loss

                # ── Step 5: Backward ──────────────────────────────────────
                self.tiktok.tik()
                self.model.backward(total_loss)
                self.model.step()
                self.tiktok.tok('backward_step')

                # ── Step 6: Collect real .grad buffers ────────────────────
                self._accumulate_real_grad(params_and_names)

                # ── Step 7: Vis log (rank-0, task > 0) ───────────────────
                if (self.args.global_rank == 0
                        and task_id > 0
                        and tmp_rounds % self.vis_interval == 0
                        and all_prev_grads_cpu is not None):

                    self._vis_log.append({
                        "step":  step,
                        "epoch": epoch,
                        "orth_norm_per_layer": proj_norms,
                        "orth_loss": orth_loss.item()
                                     if torch.is_tensor(orth_loss) else 0.0,
                        # grad_mask: (D, K) projection norms — same as full model
                        "grad_mask_per_layer": self._grad_soft_mask(
                            g_t.detach().cpu(), all_prev_grads_cpu),
                    })

                # ── Progress bar ──────────────────────────────────────────
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Epoch {epoch+1}, Step {step}, "
                        f"CE: {loss_ce.item():.4f}, "
                        f"Orth: {orth_loss.item() if torch.is_tensor(orth_loss) else 0.0:.4f}",
                        refresh=False)

                if self.args.global_rank == 0 and tmp_rounds % 30 == 0:
                    self.tiktok.print_time()

        # ── End of all epochs for this task ──────────────────────────────
        self._finalize_task_grad(task_id)
        # Persist into all_accumulate_grads (mirrors KD_LoRA_Tree.end_task)
        self._grad_store.all_accumulate_grads[task_id] = \
            self._grad_store.current_grad

        # ── Save checkpoint ───────────────────────────────────────────────
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_id))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Saved model → {peft_model_id}', self.args.global_rank)

            vis_path = os.path.join(self.args.output_dir,
                                    f"vis_log_task{task_id}.json")
            with open(vis_path, 'w') as f:
                json.dump(self._vis_log, f, indent=2)
            print_rank_0(f'Saved vis log → {vis_path}', self.args.global_rank)

    # ------------------------------------------------------------------
    # Visualisation helper  (subset of Tree_LoRA_Ortho._grad_soft_mask)
    # ------------------------------------------------------------------

    @staticmethod
    def _grad_soft_mask(g_t_cpu, all_prev_grads_cpu):
        """
        |<g_t_d, v_{k,d}>| / ||v_{k,d}||  per (layer d, prev-task k).
        Returns (D, K) list-of-lists (JSON-serialisable).
        """
        K, D, _ = all_prev_grads_cpu.shape
        mask = []
        for d in range(D):
            row = []
            for k in range(K):
                v = all_prev_grads_cpu[k, d]
                v_norm_sq = torch.dot(v, v).item()
                if v_norm_sq < 1e-12:
                    row.append(0.0)
                else:
                    proj = abs(torch.dot(g_t_cpu[d], v).item()) / (v_norm_sq ** 0.5)
                    row.append(float(proj))
            mask.append(row)
        return mask

    # ------------------------------------------------------------------
    # save_model  (mirrors Tree_LoRA convention)
    # ------------------------------------------------------------------

    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(round))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)

            adapter_config_path = os.path.join(peft_model_id, 'adapter_config.json')
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            adapter_config['r_sum'] = 0
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f)

            print_rank_0(f'Saved model → {peft_model_id}', self.args.global_rank)