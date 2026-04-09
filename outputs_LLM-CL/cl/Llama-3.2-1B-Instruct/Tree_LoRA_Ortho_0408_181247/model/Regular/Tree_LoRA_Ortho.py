import json
import os
import math
import torch
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.kd_lora_tree_ortho import KD_LoRA_Tree_Ortho
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device


class Tree_LoRA_Ortho(CL_Base_Model):
    """
    Tree_LoRA + Orthogonal Projection Loss (mathematically correct).

    ═══════════════════════════════════════════════════════════════
    MATHEMATICAL GOAL:
        Penalise current gradient g_t for aligning with past
        gradient subspace {v_k}:

            L_orth = λ/(K·D) · Σ_k Σ_d  <g_t_d, v_k_d>² / ‖v_k_d‖²

        where g_t = ∇_θ L_CE  (true gradient, not param values).

    IMPLEMENTATION STRATEGY — "Stop-Gradient Projection" (Approach B):
        The key insight: we CANNOT differentiate through g_t when
        g_t is computed with create_graph=False (cheap). Instead:

        Let α_d = <g_t_d, v_k_d> / ‖v_k_d‖²   (scalar, detached)

        Then:  d/dθ [ α_d · <θ_d, v_k_d> ]
             = α_d · v_k_d                      (gradient w.r.t. θ)

        This is equivalent to adding a penalty that PUSHES θ away
        from v_k_d, scaled by how much g_t already aligns with v_k_d.
        The direction is correct; the magnitude is the current alignment.

        This is the same trick used in:
        - GradOrth (Farajtabar et al., 2020)
        - OGD (Farajtabar et al., 2020)
        - A-GEM projected gradient

    ALTERNATIVE (Approach A, exact but expensive):
        Use create_graph=True then compute:
            g_t = autograd.grad(loss, params, create_graph=True)
            dot = sum(g_t_d * v_k_d)
            L_orth = dot² / v_norm_sq
        → backward through g_t → 2nd-order Hessian-vector products.
        Too expensive for large LoRA, often unstable with bfloat16.
    ═══════════════════════════════════════════════════════════════
    """

    def __init__(self,
                 model, tokenizer, optimizer,
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

        args.num_tasks = len(self.train_task_list)

        self.lambda_orth  = getattr(args, 'reg_orth', 0.1)
        self.vis_interval = getattr(args, 'vis_interval', 50)

        # use_second_order: exact but expensive (Approach A)
        # False = stop-gradient projection (Approach B, recommended)
        self.use_second_order = getattr(args, 'orth_second_order', False)

        self.kd_lora_tree = KD_LoRA_Tree_Ortho(args)
        self.vis_log      = []

        self._grad_accumulator = None
        self._grad_accum_count = 0

    # ──────────────────────────────────────────────────────────────────
    # param helpers
    # ──────────────────────────────────────────────────────────────────

    def _loranew_params(self):
        """Ordered list of loranew_A params with requires_grad=True."""
        return [p for n, p in self.model.named_parameters()
                if "loranew_A" in n and p.requires_grad]

    # ──────────────────────────────────────────────────────────────────
    # gradient collection
    # ──────────────────────────────────────────────────────────────────

    def _get_grad_from_loss(self, loss, create_graph=False):
        """
        Compute ∇_θ L w.r.t. loranew_A params via autograd.grad.

        Args:
            create_graph: True  → 2nd-order capable (Approach A)
                          False → detached, cheap (Approach B)

        Returns:
            list of grad tensors (same order as _loranew_params)
            OR None on failure.
        """
        params = self._loranew_params()
        if not params:
            return None, None
        try:
            grads = torch.autograd.grad(
                loss, params,
                retain_graph=True,        # keep graph for main backward
                create_graph=create_graph,
                allow_unused=True
            )
        except Exception as e:
            print_rank_0(f"autograd.grad failed: {e}", self.args.global_rank)
            return None, None

        # Return both raw grads (for 2nd-order) and stacked detached vector
        grads_clean = []
        for g, p in zip(grads, params):
            grads_clean.append(
                g if g is not None
                else torch.zeros_like(p)
            )

        # stacked: (lora_depth, D) — always detached for logging/tree
        stacked = torch.stack(
            [g.detach().reshape(-1) for g in grads_clean], dim=0
        )
        return grads_clean, stacked   # (list for 2nd-order, tensor for tree)

    def _collect_grad_after_backward(self):
        """
        Collect actual .grad buffers AFTER backward().
        Used for: gradient accumulation across epoch, visualisation.
        Returns: (lora_depth, D) or None.
        """
        params = self._loranew_params()
        if not params:
            return None
        vecs = []
        for p in params:
            vecs.append(
                p.grad.detach().reshape(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.device)
            )
        return torch.stack(vecs, dim=0)

    def _accumulate_grad(self, grad_vec):
        if self._grad_accumulator is None:
            self._grad_accumulator = grad_vec.clone()
            self._grad_accum_count = 1
        else:
            self._grad_accumulator += grad_vec
            self._grad_accum_count += 1

    # ──────────────────────────────────────────────────────────────────
    # APPROACH B: Stop-Gradient Orthogonal Loss (recommended)
    # ──────────────────────────────────────────────────────────────────

    def _orth_loss_stopgrad(self, g_t_stacked, task_id):
        """
        Stop-Gradient Projection Loss.

        For each previous task k, layer d:
            α_{k,d} = <g_t_d, v_k_d> / ‖v_k_d‖²   ← DETACHED scalar
            loss += α_{k,d} · <param_d, v_k_d>       ← DIFFERENTIABLE

        Gradient of this w.r.t. param_d:
            ∂/∂param_d = α_{k,d} · v_k_d

        Interpretation: push param in direction -α·v_k, proportional
        to how much the CURRENT gradient aligns with past gradient v_k.
        This is the GradOrth / projected-gradient trick.

        Args:
            g_t_stacked: (lora_depth, D) detached — from autograd.grad
            task_id: int
        Returns:
            orth_loss: scalar differentiable Tensor
            log_norms: list[list[float]]  [prev_k][depth_d]  for vis
        """
        eps        = 1e-8
        lora_depth = g_t_stacked.shape[0]
        orth_loss  = torch.tensor(0.0, device=self.device)
        log_norms  = []

        prev_grads = self.kd_lora_tree.all_accumulate_grads[:task_id]
        K = sum(1 for g in prev_grads if g is not None)
        if K == 0:
            return orth_loss, log_norms

        params = self._loranew_params()

        for v_k in prev_grads:
            if v_k is None:
                continue
            v_k_dev = v_k.to(self.device, non_blocking=True).detach()
            norms_k = []

            for d in range(lora_depth):
                g_d       = g_t_stacked[d]           # detached (D,)
                v_d       = v_k_dev[d]               # detached (D,)
                v_norm_sq = torch.dot(v_d, v_d) + eps

                # α = projection scalar — DETACHED, no gradient through g_t
                alpha = (torch.dot(g_d, v_d) / v_norm_sq).detach()

                # differentiable term: α · <param_d, v_d>
                # ∂/∂param_d = α · v_d  → pushes param away from v_d
                #              when α > 0 (gradient aligns with past task)
                p_d      = params[d].reshape(-1)     # differentiable (D,)
                orth_loss = orth_loss + alpha * torch.dot(p_d, v_d)

                # for logging: compute proj norm using g_t
                proj_norm = (alpha.item() ** 2 * v_norm_sq.item()) ** 0.5
                norms_k.append(proj_norm)

            log_norms.append(norms_k)

        # scale: lambda / (K * D)
        orth_loss = self.lambda_orth * orth_loss / (K * lora_depth)
        return orth_loss, log_norms

    # ──────────────────────────────────────────────────────────────────
    # APPROACH A: True 2nd-order Orthogonal Loss (exact but expensive)
    # ──────────────────────────────────────────────────────────────────

    def _orth_loss_second_order(self, grads_list, task_id):
        """
        Exact orthogonal loss via 2nd-order gradients.

        Requires grads_list from autograd.grad(..., create_graph=True).

            L_orth = λ/(K·D) · Σ_k Σ_d  <g_t_d, v_k_d>² / ‖v_k_d‖²

        Differentiating <g_t_d, v_k_d>² w.r.t. θ triggers Hessian-
        vector products internally. Memory: ~2x forward pass.

        ⚠️  Can be unstable with bfloat16. Recommend float32 if using this.
        """
        eps        = 1e-8
        lora_depth = len(grads_list)
        orth_loss  = torch.tensor(0.0, device=self.device)
        log_norms  = []

        prev_grads = self.kd_lora_tree.all_accumulate_grads[:task_id]
        K = sum(1 for g in prev_grads if g is not None)
        if K == 0:
            return orth_loss, log_norms

        for v_k in prev_grads:
            if v_k is None:
                continue
            v_k_dev = v_k.to(self.device, non_blocking=True).detach()
            norms_k = []

            for d in range(lora_depth):
                g_d       = grads_list[d].reshape(-1)   # differentiable!
                v_d       = v_k_dev[d]
                v_norm_sq = torch.dot(v_d, v_d) + eps

                dot_val  = torch.dot(g_d, v_d)          # differentiable
                proj_sq  = dot_val ** 2 / v_norm_sq     # differentiable
                orth_loss = orth_loss + proj_sq

                norms_k.append(proj_sq.detach().item() ** 0.5)

            log_norms.append(norms_k)

        orth_loss = self.lambda_orth * orth_loss / (K * lora_depth)
        return orth_loss, log_norms

    # ──────────────────────────────────────────────────────────────────
    # dispatch
    # ──────────────────────────────────────────────────────────────────

    def _orthogonal_projection_loss(self, grads_list, g_t_stacked, task_id):
        """
        Dispatch to correct approach based on self.use_second_order.

        Args:
            grads_list : list of grad Tensors from autograd.grad
                         (with create_graph=True if use_second_order)
            g_t_stacked: (lora_depth, D) detached stacked version
            task_id    : int
        """
        if self.use_second_order:
            return self._orth_loss_second_order(grads_list, task_id)
        else:
            return self._orth_loss_stopgrad(g_t_stacked, task_id)

    # ──────────────────────────────────────────────────────────────────
    # visualisation (uses REAL .grad after backward)
    # ──────────────────────────────────────────────────────────────────

    def _record_vis(self, task_id, epoch, step, real_grad):
        """
        real_grad: (lora_depth, D) from .grad buffers after backward.
        This is the TRUE gradient that was actually used to update params.
        """
        if real_grad is None:
            return
        lora_depth = real_grad.shape[0]
        eps = 1e-8
        cos_records, orth_records = [], []

        for v_k in self.kd_lora_tree.all_accumulate_grads[:task_id]:
            if v_k is None:
                continue
            v_k_dev = v_k.to(self.device, non_blocking=True).detach()
            cos_k, orth_k = [], []

            for d in range(lora_depth):
                g_d    = real_grad[d]
                v_d    = v_k_dev[d]
                g_norm = torch.norm(g_d).item()
                v_norm = torch.norm(v_d).item()

                # cosine similarity with REAL gradient
                cos_val = (torch.dot(g_d, v_d) /
                           (g_norm * v_norm + eps)).item()

                # projection norm
                v_norm_sq = v_norm ** 2 + eps
                proj_norm = ((torch.dot(g_d, v_d) ** 2 /
                              v_norm_sq).item() ** 0.5)

                cos_k.append(cos_val)
                orth_k.append(proj_norm)

            cos_records.append(cos_k)
            orth_records.append(orth_k)

        self.vis_log.append(dict(
            task_id=task_id, epoch=epoch, step=step,
            cos_sim_per_layer=cos_records,
            orth_norm_per_layer=orth_records,
        ))

    # ──────────────────────────────────────────────────────────────────
    # main training loop
    # ──────────────────────────────────────────────────────────────────

    def train_one_task(self, task, task_id, epochs):
        train_dataloader     = self.train_task_list[task]
        eval_dataloader      = self.eval_task_list[task]
        total_steps          = epochs * len(train_dataloader)
        train_dataloader_len = len(train_dataloader)

        progress_bar = tqdm(
            total=total_steps, leave=True,
            disable=(self.args.global_rank != 0)
        )

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, "
                f"Total Micro Batches {train_dataloader_len}",
                self.args.global_rank
            )
            self.model.train()
            self.tiktok.print_time(self.args.global_rank)

            self.kd_lora_tree.new_epoch_init(train_dataloader_len)
            self._grad_accumulator = None
            self._grad_accum_count = 0
            tmp_rounds = -1

            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1

                if self.args.reg > 0:
                    self.kd_lora_tree.step()

                del batch['sources']
                batch   = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss_ce = outputs.loss

                reg_loss  = torch.tensor(0.0, device=self.device)
                orth_loss = torch.tensor(0.0, device=self.device)

                if self.args.reg > 0:
                    # ── Get instantaneous gradient ──────────────────────
                    # Approach A needs create_graph=True (2nd-order)
                    # Approach B needs create_graph=False (cheaper)
                    self.tiktok.tik()
                    need_graph = self.use_second_order and task_id > 0
                    grads_list, g_t_stacked = self._get_grad_from_loss(
                        loss_ce, create_graph=need_graph
                    )
                    self.tiktok.tok(f"autograd_@T{task_id}_E{epoch}")

                    if g_t_stacked is not None:
                        # insert into KD-tree (always uses detached stacked)
                        self.kd_lora_tree.insert_grad(g_t_stacked)

                        if task_id > 0:
                            # ── Tree similarity regulariser (unchanged) ──
                            self.tiktok.tik()
                            prev_id_matrix = self.kd_lora_tree.tree_search(
                                task_id, device=self.device)
                            reg_loss = self.kd_lora_tree.get_loss(
                                g_t_stacked, loss_ce, task_id, prev_id_matrix)
                            self.tiktok.tok(f"tree_reg_@T{task_id}_E{epoch}")

                            # ── Orthogonal Projection Loss ───────────────
                            self.tiktok.tik()
                            orth_loss, _ = self._orthogonal_projection_loss(
                                grads_list, g_t_stacked, task_id)
                            self.tiktok.tok(f"orth_@T{task_id}_E{epoch}")

                            if tmp_rounds % 100 == 0:
                                approach = "2nd-order" if self.use_second_order \
                                           else "stop-grad"
                                print_rank_0(
                                    f"\033[34m[{approach}] "
                                    f"Reg: {reg_loss.item():.4f} | "
                                    f"Orth: {orth_loss.item():.4f}\033[0m",
                                    self.args.global_rank
                                )

                # total loss
                loss = loss_ce - reg_loss + orth_loss

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Epoch {epoch+1}, Step {step}, "
                        f"Loss: {loss.item():.4f}", refresh=False)

                # ── backward ────────────────────────────────────────────
                self.tiktok.tik()
                self.model.backward(loss)

                # ── collect REAL .grad after backward (for vis & accum) ─
                real_grad = self._collect_grad_after_backward()
                if real_grad is not None:
                    self._accumulate_grad(real_grad)

                    if task_id > 0 and tmp_rounds % self.vis_interval == 0:
                        self._record_vis(task_id, epoch, step, real_grad)
                        if self.vis_log:
                            self.vis_log[-1]['orth_loss'] = orth_loss.item()

                self.model.step()
                self.tiktok.tok('backward+step')

                if self.args.global_rank == 0 and tmp_rounds % 30 == 0:
                    self.tiktok.print_time()

            # ── end of epoch: commit mean grad to KD-tree ───────────────
            if self._grad_accumulator is not None:
                self.kd_lora_tree.current_grad = (
                    self._grad_accumulator / self._grad_accum_count
                )

        # ── save model ──────────────────────────────────────────────────
        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_id))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Saved → {peft_model_id}', self.args.global_rank)

            vis_path = os.path.join(
                self.args.output_dir, f'vis_log_task{task_id}.json')
            with open(vis_path, 'w') as f:
                json.dump(self.vis_log, f, indent=2)
            print_rank_0(f'Vis log → {vis_path}', self.args.global_rank)

        if self.args.reg > 0:
            self.kd_lora_tree.end_task(task_id=task_id)

    def save_model(self, round):
        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(round))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            adapter_config_path = os.path.join(
                peft_model_id, 'adapter_config.json')
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            adapter_config['r_sum'] = 0
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f)