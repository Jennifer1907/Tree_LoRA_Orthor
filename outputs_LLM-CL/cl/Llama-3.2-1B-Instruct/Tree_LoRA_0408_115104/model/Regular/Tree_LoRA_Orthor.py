"""
Tree_LoRA_Orthor.py  —  Tree-LoRA + OPL continual learning trainer  (fixed v2)

Critical fixes vs original
---------------------------
1.  NO second backward for reg loss.
    apply_grad_surgery() modifies param.grad IN-PLACE after CE backward.
    A second model.backward(reg_loss) under DeepSpeed zeros then re-populates
    .grad, silently destroying the CE gradients that insert_grad() already
    collected — this was the PRIMARY cause of current_grad=None at end_task().

2.  insert_grad() is UNCONDITIONALLY called every step, immediately after
    _collect_lora_grads(), before anything else touches .grad.

3.  _collect_lora_params() removed — gradient surgery operates directly on
    .grad; param values are never needed for the reg term.

4.  Startup diagnostic: _debug_lora_params() prints requires_grad and name
    of every loranew_A parameter once at construction time.

5.  Progress bar shows running total_grad_steps so you can confirm
    accumulation is working without waiting for end_task().

Correct per-step order
----------------------
    forward  →  ce_loss
    model.backward(ce_loss)          ← .grad populated
    _collect_lora_grads()            ← snapshot detached .grad (loranew_A only)
    insert_grad(grad_features)       ← accumulate into current_grad (ALWAYS)
    [optional: debug logging]
    if task_id > 0 and reg > 0:
        tree_search()                ← prev_id_matrix (LOCAL indices)
        _get_deflated_opl_bases()    ← per-depth constant bases (CPU)
        apply_grad_surgery()         ← modify .grad in-place (NO backward)
    model.step()                     ← reads modified .grad, zeros it
"""

import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.base_model import CL_Base_Model
from utils.kd_lora_tree_Orthor import (
    KD_LoRA_Tree,
    _build_deflated_opl_basis,
    apply_grad_surgery,
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
        - Tree-LoRA gradient alignment  (in-place .grad modification)
        - Orthogonal Projection Loss    (in-place .grad modification)
        - KD-tree bandit task selection

    The regularisation is applied via gradient surgery after CE backward —
    NO second backward pass is used.

    Parameters
    ----------
    lamda_1, lamda_2 : kept for API compatibility with other CL_Base_Model
                       subclasses; lamda_opl is read from args.
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

        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

        num_task       = len(self.train_task_list)
        args.num_tasks = num_task

        # Alignment strength used inside apply_grad_surgery.
        # Stored on the model object so apply_grad_surgery can read it.
        # alpha > 0 amplifies the CE gradient along the alignment direction.
        # alpha = 0 leaves the CE gradient unchanged along that direction.
        # alpha = -1 removes the component entirely (pure GEM).
        self.model._tree_lora_alpha = getattr(args, "tree_lora_alpha", 0.5)

        self.kd_lora_tree = KD_LoRA_Tree(args)

        self.viz_interval = getattr(args, "viz_interval", 200)
        self.viz_dir      = os.path.join(getattr(args, "output_dir", "."), "viz")

        self.grad_snapshots: List[Optional[torch.Tensor]] = [None] * num_task

        # Startup check — print loranew_A param status once
        if self.args.global_rank == 0:
            self._debug_lora_params()

    # ------------------------------------------------------------------
    # Startup diagnostics
    # ------------------------------------------------------------------

    def _debug_lora_params(self) -> None:
        """
        Print trainability status of every loranew_A parameter once at init.
        Catches mis-named params, frozen params, and missing PEFT setup early.
        """
        found = []
        for name, param in self.model.named_parameters():
            if "loranew_A" in name:
                found.append((name, param.requires_grad, tuple(param.shape)))

        if not found:
            print_rank_0(
                "\033[31m[CRITICAL] No parameters named 'loranew_A' found.\n"
                "  → Check that get_peft_model() was called with the correct\n"
                "    LoraConfig and that the PEFT library creates loranew_A weights.\n"
                "  → Try: for n,p in model.named_parameters(): print(n)\033[0m",
                self.args.global_rank,
            )
            return

        non_trainable = [(n, s) for n, rg, s in found if not rg]
        print_rank_0(
            f"\033[34m[init_debug] {len(found)} loranew_A params found, "
            f"{len(found) - len(non_trainable)} trainable.\n"
            f"  First 3: {found[:3]}\033[0m",
            self.args.global_rank,
        )
        if non_trainable:
            print_rank_0(
                f"\033[31m[WARN] {len(non_trainable)} loranew_A params have "
                f"requires_grad=False — these will never produce .grad!\n"
                f"  First 3: {non_trainable[:3]}\033[0m",
                self.args.global_rank,
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_task(self, task, task_id: int, epochs: int):
        """
        Fixed training loop.

        Key invariant
        -------------
        insert_grad() is called EVERY step that produces a non-None
        grad_features.  It must be called before model.step() because
        DeepSpeed's model.step() calls zero_grad() internally.

        Regularisation is applied by apply_grad_surgery(), which modifies
        param.grad in-place.  model.step() then reads the modified gradients.
        There is NO second model.backward() call.
        """
        train_dataloader     = self.train_task_list[task]
        total_steps          = epochs * len(train_dataloader)
        train_dataloader_len = len(train_dataloader)
        progress_bar = tqdm(
            total=total_steps, leave=True,
            disable=(self.args.global_rank != 0),
        )

        # Reset gradient accumulator ONCE per task (not per epoch).
        # current_grad must accumulate across ALL epochs of the task.
        self.kd_lora_tree.new_task_init()

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Epoch {epoch + 1}/{epochs} — {train_dataloader_len} micro-batches",
                self.args.global_rank,
            )
            self.model.train()
            self.tiktok.print_time(self.args.global_rank)

            # Reset per-epoch bandit state (NOT current_grad — see new_epoch_init docs)
            self.kd_lora_tree.new_epoch_init(train_dataloader_len)
            tmp_rounds = -1

            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1
                self.kd_lora_tree.step()   # advance ramp counter

                batch.pop("sources", None)
                batch = to_device(batch, self.device)

                # ── 1. Forward ─────────────────────────────────────────
                outputs = self.model(**batch, use_cache=False)
                ce_loss = outputs.loss

                # ── 2. CE backward ─────────────────────────────────────
                # After this call, loranew_A.grad is populated (if the
                # parameter is used in the forward pass and requires_grad=True).
                self.tiktok.tik()
                self.model.backward(ce_loss)
                self.tiktok.tok("ce_backward")

                # ── 3. Collect gradients and ALWAYS insert ──────────────
                # This MUST happen before model.step(), which calls zero_grad().
                # insert_grad() is UNCONDITIONAL — current_grad must accumulate
                # even when task_id == 0 and no reg is applied.
                self.tiktok.tik()
                _grad_features = self._collect_lora_grads()
                self.tiktok.tok(f"Grad_Collect@T{task_id}E{epoch}")

                if _grad_features is not None:
                    # ALWAYS accumulate — needed for end_task() KD-tree / OPL build
                    self.kd_lora_tree.insert_grad(_grad_features)
                else:
                    # Diagnose on the first occurrence each epoch
                    if tmp_rounds == 0 and self.args.global_rank == 0:
                        print_rank_0(
                            f"\033[31m[WARN] T{task_id} E{epoch} S{tmp_rounds}: "
                            f"_collect_lora_grads() returned None.\n"
                            f"Detailed loranew_A .grad status:\033[0m",
                            self.args.global_rank,
                        )
                        for name, param in self.model.named_parameters():
                            if "loranew_A" in name:
                                print_rank_0(
                                    f"  {name}  requires_grad={param.requires_grad}  "
                                    f"grad_is_none={param.grad is None}",
                                    self.args.global_rank,
                                )

                # Periodic health log
                if (
                    _grad_features is not None
                    and tmp_rounds % 100 == 0
                    and self.args.global_rank == 0
                ):
                    gnorms = _grad_features.norm(dim=1)
                    n_zero = (gnorms < 1e-8).sum().item()
                    print_rank_0(
                        f"\033[33m[grad_debug] T{task_id} S{tmp_rounds} "
                        f"mean_norm={gnorms.mean().item():.4e}  "
                        f"zero_depths={n_zero}/{_grad_features.shape[0]}  "
                        f"total_grad_steps={self.kd_lora_tree.total_grad_steps}\033[0m",
                        self.args.global_rank,
                    )

                # ── 4. Gradient surgery (task_id > 0 only) ──────────────
                # apply_grad_surgery() modifies param.grad in-place.
                # NO second backward — that was the root cause of the bug.
                if (
                    self.args.reg > 0
                    and task_id > 0
                    and _grad_features is not None
                    and self.kd_lora_tree.tmp_reg > 0
                ):
                    self.tiktok.tik()
                    prev_id_matrix = self.kd_lora_tree.tree_search(
                        task_id, device=self.device
                    )
                    self.tiktok.tok(f"TreeSearch@T{task_id}E{epoch}")

                    if self.kd_lora_tree.all_grad is not None:
                        deflated_bases = self.kd_lora_tree._get_deflated_opl_bases(
                            task_id, prev_id_matrix
                        )

                        apply_grad_surgery(
                            model          = self.model,
                            all_grad       = self.kd_lora_tree.all_grad,
                            prev_id_matrix = prev_id_matrix,
                            opl_bases      = deflated_bases,
                            lambda_opl     = (
                                self.kd_lora_tree.lambda_opl
                                * self.kd_lora_tree.tmp_reg
                            ),
                            param_keyword  = "loranew_A",
                        )

                        if tmp_rounds % 100 == 0 and self.args.global_rank == 0:
                            print_rank_0(
                                f"\033[34m[reg_debug] T{task_id} S{tmp_rounds}  "
                                f"CE={ce_loss.item():.4f}  "
                                f"reg_scale={self.kd_lora_tree.tmp_reg:.4f}  "
                                f"lambda_opl={self.kd_lora_tree.lambda_opl:.3f}  "
                                f"prev_ids={prev_id_matrix.tolist()}\033[0m",
                                self.args.global_rank,
                            )

                # ── Visualisation snapshot ──────────────────────────────
                if (
                    self.args.global_rank == 0
                    and tmp_rounds % self.viz_interval == 0
                    and _grad_features is not None
                ):
                    self._maybe_save_viz_snapshot(task_id, tmp_rounds, epoch)

                # ── 5. Optimiser step ───────────────────────────────────
                # DeepSpeed model.step() reads param.grad (now modified by
                # surgery if applicable) then zeroes it internally.
                self.model.step()

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"T{task_id} E{epoch+1} S{step} "
                        f"ce={ce_loss.item():.4f}  "
                        f"grad_steps={self.kd_lora_tree.total_grad_steps}",
                        refresh=False,
                    )
                if self.args.global_rank == 0 and tmp_rounds % 30 == 0:
                    self.tiktok.print_time()

        # ── Post-task ──────────────────────────────────────────────────
        self._save_checkpoint(task_id)
        self.kd_lora_tree.end_task(task_id=task_id)

        if self.kd_lora_tree.current_grad is not None:
            self.grad_snapshots[task_id] = self.kd_lora_tree.current_grad.clone()

        if self.args.global_rank == 0 and task_id >= 1:
            self._generate_post_task_heatmaps(task_id)

    # ------------------------------------------------------------------
    # Gradient collection
    # ------------------------------------------------------------------

    def _collect_lora_grads(self) -> Optional[torch.Tensor]:
        """
        Snapshot .grad of every loranew_A parameter after CE backward().

        Returns (lora_depth, D) GPU tensor — fully detached, requires_grad=False.
        Returns None if no loranew_A parameter has a non-None .grad.

        CALL TIMING: immediately after model.backward(ce_loss),
                     before model.step() (which zeros .grad).
        """
        parts = []
        for name, param in self.model.named_parameters():
            if "loranew_A" in name and param.grad is not None:
                parts.append(param.grad.detach().reshape(-1))  # (D_i,)

        if not parts:
            return None

        max_len = max(p.shape[0] for p in parts)
        if any(p.shape[0] != max_len for p in parts):
            parts = [
                F.pad(p, (0, max_len - p.shape[0])) if p.shape[0] < max_len else p
                for p in parts
            ]

        return torch.stack(parts, dim=0)   # (lora_depth, D)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, task_id: int):
        if self.args.output_dir is None or self.args.global_rank != 0:
            return
        peft_model_id = os.path.join(self.args.output_dir, str(task_id))
        os.makedirs(peft_model_id, exist_ok=True)
        self.model.save_pretrained(peft_model_id)
        self.tokenizer.save_pretrained(peft_model_id)
        print_rank_0(f"Checkpoint saved → {peft_model_id}", self.args.global_rank)

    def save_model(self, round: int):
        if self.args.output_dir is None or self.args.global_rank != 0:
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
        snap = collect_gradient_snapshot(self.model, param_keyword="loranew_A")
        if snap is None:
            return
        os.makedirs(self.viz_dir, exist_ok=True)
        path = os.path.join(
            self.viz_dir, f"grad_snap_t{task_id}_e{epoch}_s{step}.pt"
        )
        torch.save(snap, path)

    def _generate_post_task_heatmaps(self, task_id: int):
        valid = [
            (t, g)
            for t, g in enumerate(self.grad_snapshots[: task_id + 1])
            if g is not None
        ]
        if len(valid) < 2:
            return

        task_ids   = [t for t, _ in valid]
        all_grads  = torch.stack([g for _, g in valid], dim=0)
        labels     = [f"T{t}" for t in task_ids]
        lora_depth = all_grads.shape[1]

        os.makedirs(self.viz_dir, exist_ok=True)

        plot_gradient_similarity_heatmap(
            all_grads,
            task_labels=labels,
            save_path=os.path.join(self.viz_dir, "cosine_sim_depth{}.png"),
        )

        opl_bases_deflated_per_task = []
        for t in task_ids:
            raw_bases = (
                self.kd_lora_tree.opl_basis[t]
                if t < len(self.kd_lora_tree.opl_basis) and self.kd_lora_tree.opl_basis[t]
                else [None] * lora_depth
            )
            if not raw_bases:
                raw_bases = [None] * lora_depth

            prev_for_t = [
                (pt, self.grad_snapshots[pt])
                for pt in range(t)
                if self.grad_snapshots[pt] is not None
            ]
            if not prev_for_t:
                opl_bases_deflated_per_task.append(raw_bases)
                continue

            prev_grads_t = torch.stack([g for _, g in prev_for_t], dim=0)
            cg_t         = self.grad_snapshots[t]

            deflated = []
            for d in range(lora_depth):
                B_raw = raw_bases[d] if d < len(raw_bases) else None
                if B_raw is None or cg_t is None:
                    deflated.append(B_raw)
                    continue
                pg_d   = prev_grads_t[:, d, :]
                cg_d   = cg_t[d]
                cg_d_n = F.normalize(cg_d.unsqueeze(0), dim=1)
                pg_d_n = F.normalize(pg_d, dim=1)
                cos_d  = (pg_d_n @ cg_d_n.T).squeeze(1)
                best_p = int(cos_d.argmax().item())
                a_dir  = pg_d[best_p]
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

        current_grad = self.grad_snapshots[task_id]
        if current_grad is not None and len(valid) >= 2:
            prev_valid = [(t, g) for t, g in valid if t < task_id]
            if prev_valid:
                prev_grads_only = torch.stack([g for _, g in prev_valid], dim=0)
                bases_for_c     = (
                    opl_bases_deflated_per_task[-1]
                    if opl_bases_deflated_per_task
                    else [None] * lora_depth
                )
                metrics = compute_projection_metrics(
                    current_grad, prev_grads_only, bases_for_c
                )
                plot_opl_effect_heatmap(
                    metrics,
                    task_id=task_id,
                    save_dir=self.viz_dir,
                )