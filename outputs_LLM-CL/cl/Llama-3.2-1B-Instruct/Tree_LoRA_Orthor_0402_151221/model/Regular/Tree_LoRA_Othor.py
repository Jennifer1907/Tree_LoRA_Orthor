"""
tree_lora.py  —  Revised Tree-LoRA + OPL continual learning trainer

Changes from original
---------------------
- Uses the revised KD_LoRA_Tree (local-index KD-tree, combined L_sim + L_opl)
- Gradient collection uses only live loranew_A parameters, no None-stack
- Visualisation hooks: snapshot collected every `args.viz_interval` steps
  and written to `args.output_dir/viz/`
- Safe first-task and task_id==0 paths
- DeepSpeed-compatible (single-GPU mode when local_rank == -1)
"""

import json
import os
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.base_model import CL_Base_Model
from utils.kd_lora_tree import (
    KD_LoRA_Tree,
    _build_deflated_opl_basis,
    collect_gradient_snapshot,
    compute_projection_metrics,
    plot_gradient_similarity_heatmap,
    plot_projection_heatmap,
    plot_opl_effect_heatmap,
)
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device


class Tree_LoRA(CL_Base_Model):
    """
    Continual learning trainer combining:
        - Tree-LoRA gradient alignment (similarity regularisation)
        - Orthogonal Projection Loss (OPL) over previous-task subspaces
        - KD-tree bandit task selection

    The combined regulariser is:
        L_reg = L_sim  +  lambda_opl * L_opl

    L_sim  : cosine alignment with the bandit-selected previous task per depth
    L_opl  : squared projection ratio onto the subspace of ALL OTHER previous
             tasks (excluding the alignment direction so the objectives do not
             cancel each other — see kd_lora_tree.py for details)
    """

    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        train_task_list,
        eval_task_list,
        test_task_list,
        args,
        lamda_1: float = 0.5,
        lamda_2: float = 0.0,
    ):
        super().__init__(
            model, tokenizer, optimizer,
            train_task_list, eval_task_list, test_task_list, args,
        )
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.tiktok  = TIKTOK(args)

        # Device setup
        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

        num_task = len(self.train_task_list)
        args.num_tasks = num_task

        self.kd_lora_tree = KD_LoRA_Tree(args)

        # Visualisation settings
        self.viz_interval = getattr(args, "viz_interval", 200)   # steps between snapshots
        self.viz_dir = os.path.join(getattr(args, "output_dir", "."), "viz")

        # CPU-side gradient snapshot store for visualisations
        # grad_snapshots[task_id] = (lora_depth, D) tensor, CPU
        self.grad_snapshots: List[Optional[torch.Tensor]] = [None] * num_task

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_task(self, task, task_id: int, epochs: int):
        train_dataloader = self.train_task_list[task]
        eval_dataloader  = self.eval_task_list[task]

        total_steps          = epochs * len(train_dataloader)
        train_dataloader_len = len(train_dataloader)
        progress_bar = tqdm(
            total=total_steps, leave=True,
            disable=(self.args.global_rank != 0),
        )

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Epoch {epoch + 1}/{epochs} — {train_dataloader_len} micro-batches",
                self.args.global_rank,
            )
            self.model.train()
            self.tiktok.print_time(self.args.global_rank)

            self.kd_lora_tree.new_epoch_init(train_dataloader_len)
            tmp_rounds = -1

            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1

                if self.args.reg > 0:
                    self.kd_lora_tree.step()

                batch.pop("sources", None)
                batch = to_device(batch, self.device)

                outputs = self.model(**batch, use_cache=False)
                loss    = outputs.loss

                # ---- regularisation ----
                if self.args.reg > 0:
                    self.tiktok.tik()

                    # Collect live loranew_A parameters (no grad needed here)
                    _grad_parts = self._collect_lora_params()

                    if _grad_parts is not None:
                        # Stack into (lora_depth, D)  — stays on GPU
                        _grad_current = _grad_parts   # (lora_depth, D)

                        # Accumulate into running mean (CPU)
                        self.kd_lora_tree.insert_grad(_grad_current)

                        self.tiktok.tok(f"Grad_Collect@T{task_id}E{epoch}")

                        # Tree-LoRA + OPL regularisation (task_id > 0 only)
                        if task_id > 0:
                            self.tiktok.tik()
                            prev_id_matrix = self.kd_lora_tree.tree_search(
                                task_id, device=self.device
                            )
                            self.tiktok.tok(f"TreeSearch@T{task_id}E{epoch}")

                            self.tiktok.tik()
                            reg_loss = self.kd_lora_tree.get_loss(
                                _grad_current, loss, task_id, prev_id_matrix
                            )
                            # L_reg is a non-negative penalty (L_sim >= 0, L_opl >= 0)
                            loss = loss + reg_loss
                            self.tiktok.tok(f"RegLoss@T{task_id}E{epoch}")

                            if tmp_rounds % 100 == 0:
                                print_rank_0(
                                    f"\033[34mReg={reg_loss.item():.4f}  "
                                    f"CE={outputs.loss.item():.4f}  "
                                    f"prev_ids={prev_id_matrix.tolist()}\033[0m",
                                    self.args.global_rank,
                                )

                        # Periodic visualisation snapshot
                        if (
                            self.args.global_rank == 0
                            and tmp_rounds % self.viz_interval == 0
                        ):
                            self._maybe_save_viz_snapshot(
                                task_id, tmp_rounds, epoch
                            )

                # ---- progress bar ----
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"T{task_id} E{epoch+1} S{step} loss={loss.item():.4f}",
                        refresh=False,
                    )

                # ---- backward / step ----
                self.tiktok.tik()
                self.model.backward(loss)
                self.model.step()
                self.tiktok.tok("backward")

                if self.args.global_rank == 0 and tmp_rounds % 30 == 0:
                    self.tiktok.print_time()

        # ---- post-task: save checkpoint ----
        self._save_checkpoint(task_id)

        # ---- post-task: update tree & OPL bases ----
        if self.args.reg > 0:
            self.kd_lora_tree.end_task(task_id=task_id)

        # ---- post-task: store snapshot for visualisations ----
        if self.kd_lora_tree.current_grad is not None:
            self.grad_snapshots[task_id] = self.kd_lora_tree.current_grad.clone()

        # ---- post-task: generate summary heatmaps ----
        if self.args.global_rank == 0 and task_id >= 1:
            self._generate_post_task_heatmaps(task_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_lora_params(self) -> Optional[torch.Tensor]:
        """
        Collect loranew_A parameters as a (lora_depth, D) GPU tensor.
        Returns None if no such parameters exist.
        """
        parts = []
        for name, param in self.model.named_parameters():
            if "loranew_A" in name:
                parts.append(param.reshape(-1))   # keep on GPU, still in graph
        if not parts:
            return None
        return torch.stack(parts, dim=0)           # (lora_depth, D)

    # ------------------------------------------------------------------
    def _save_checkpoint(self, task_id: int):
        if self.args.output_dir is None:
            return
        if self.args.global_rank != 0:
            return

        peft_model_id = os.path.join(self.args.output_dir, str(task_id))
        os.makedirs(peft_model_id, exist_ok=True)
        self.model.save_pretrained(peft_model_id)
        self.tokenizer.save_pretrained(peft_model_id)
        print_rank_0(f"Checkpoint saved → {peft_model_id}", self.args.global_rank)

    # ------------------------------------------------------------------
    def save_model(self, round: int):
        """Save model for a given round, resetting r_sum for O-LoRA compatibility."""
        if self.args.output_dir is None:
            return
        if self.args.global_rank != 0:
            return

        peft_model_id = os.path.join(self.args.output_dir, str(round))
        os.makedirs(peft_model_id, exist_ok=True)
        self.model.save_pretrained(peft_model_id)
        self.tokenizer.save_pretrained(peft_model_id)

        adapter_cfg_path = os.path.join(peft_model_id, "adapter_config.json")
        if os.path.isfile(adapter_cfg_path):
            with open(adapter_cfg_path, "r") as f:
                cfg = json.load(f)
            cfg["r_sum"] = 0
            with open(adapter_cfg_path, "w") as f:
                json.dump(cfg, f)

        print_rank_0(f"Model saved → {peft_model_id}", self.args.global_rank)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _maybe_save_viz_snapshot(self, task_id: int, step: int, epoch: int):
        """
        Save a gradient snapshot (from model .grad attributes) for later
        heatmap generation.  Runs only on rank 0 and only if gradients exist.
        """
        snap = collect_gradient_snapshot(self.model, param_keyword="loranew_A")
        if snap is None:
            return
        os.makedirs(self.viz_dir, exist_ok=True)
        path = os.path.join(
            self.viz_dir, f"grad_snap_t{task_id}_e{epoch}_s{step}.pt"
        )
        torch.save(snap, path)

    # ------------------------------------------------------------------
    def _generate_post_task_heatmaps(self, task_id: int):
        """
        After completing task `task_id`, generate three sets of heatmaps.
        All tensor operations run on CPU; no GPU memory is used.

        Heatmap A — cosine similarity across all seen tasks (per LoRA depth)
        Heatmap B — projection ratio onto DEFLATED OPL subspace per task/depth
        Heatmap C — per-depth OPL effect breakdown for the current task

        For heatmaps B and C we use the DEFLATED bases (not the raw pre-built
        ones) so the projection ratios reflect what OPL actually penalises.
        Because the actual alignment partner used at each step is not recorded,
        we approximate deflation using the most similar previous task per depth
        (same strategy as top-k selection) as a representative alignment dir.
        """
        valid = [
            (t, g)
            for t, g in enumerate(self.grad_snapshots[: task_id + 1])
            if g is not None
        ]
        if len(valid) < 2:
            return

        task_ids   = [t for t, _ in valid]
        all_grads  = torch.stack([g for _, g in valid], dim=0)  # (T, lora_depth, D)
        labels     = [f"T{t}" for t in task_ids]
        lora_depth = all_grads.shape[1]

        os.makedirs(self.viz_dir, exist_ok=True)

        # --- Heatmap A: cosine similarity ---
        plot_gradient_similarity_heatmap(
            all_grads,
            task_labels=labels,
            save_path=os.path.join(self.viz_dir, "cosine_sim_depth{}.png"),
        )

        # --- Heatmap B: projection strength (deflated bases) ---
        # For each task t, use its pre-built raw basis deflated by the most
        # cosine-similar previous task at each depth as the alignment direction.
        opl_bases_deflated_per_task = []
        for t in task_ids:
            raw_bases = (
                self.kd_lora_tree.opl_basis[t]
                if t < len(self.kd_lora_tree.opl_basis) and self.kd_lora_tree.opl_basis[t]
                else [None] * lora_depth
            )
            if not raw_bases:
                raw_bases = [None] * lora_depth

            # Find which previous tasks exist for task t
            prev_for_t = [
                (pt, self.grad_snapshots[pt])
                for pt in range(t)
                if self.grad_snapshots[pt] is not None
            ]
            if not prev_for_t:
                opl_bases_deflated_per_task.append(raw_bases)
                continue

            prev_grads_t = torch.stack([g for _, g in prev_for_t], dim=0)  # (np, L, D)
            cg_t = self.grad_snapshots[t]                                    # (L, D)

            deflated = []
            for d in range(lora_depth):
                B_raw = raw_bases[d] if d < len(raw_bases) else None
                if B_raw is None or cg_t is None:
                    deflated.append(B_raw)
                    continue
                # Approximate alignment direction: most cosine-similar prev task at depth d
                pg_d   = prev_grads_t[:, d, :]                              # (np, D)
                cg_d   = cg_t[d]                                            # (D,)
                cg_d_n = F.normalize(cg_d.unsqueeze(0), dim=1)
                pg_d_n = F.normalize(pg_d, dim=1)
                cos_d  = (pg_d_n @ cg_d_n.T).squeeze(1)
                best_p = int(cos_d.argmax().item())
                a_dir  = pg_d[best_p]                                       # (D,)
                B_defl = _build_deflated_opl_basis(
                    candidate_grads=B_raw,
                    alignment_dir=a_dir,
                    max_rank=self.kd_lora_tree.opl_max_rank,
                )
                deflated.append(B_defl)
            opl_bases_deflated_per_task.append(deflated)

        plot_projection_heatmap(
            all_grads,
            opl_bases_per_task=opl_bases_deflated_per_task,
            task_labels=labels,
            save_path=os.path.join(self.viz_dir, f"projection_after_task{task_id}.png"),
        )

        # --- Heatmap C: per-depth OPL effect for current task ---
        current_grad = self.grad_snapshots[task_id]
        if current_grad is not None and len(valid) >= 2:
            # Previous task gradients (exclude current task itself)
            prev_valid = [(t, g) for t, g in valid if t < task_id]
            if prev_valid:
                prev_grads_only = torch.stack([g for _, g in prev_valid], dim=0)  # (np, L, D)

                # Use deflated bases for current task (last entry in opl_bases_deflated)
                bases_for_c = opl_bases_deflated_per_task[-1] if opl_bases_deflated_per_task else [None] * lora_depth

                metrics = compute_projection_metrics(
                    current_grad, prev_grads_only, bases_for_c
                )
                plot_opl_effect_heatmap(
                    metrics,
                    task_id=task_id,
                    save_dir=self.viz_dir,
                )