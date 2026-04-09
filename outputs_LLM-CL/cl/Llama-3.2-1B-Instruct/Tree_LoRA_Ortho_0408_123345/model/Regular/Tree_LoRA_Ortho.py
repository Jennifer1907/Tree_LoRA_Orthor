import copy
import json
import os
import pickle
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.kd_lora_tree_ortho import KD_LoRA_Tree_Ortho
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


class Tree_LoRA_Ortho(CL_Base_Model):
    """
    Tree_LoRA with Orthogonal Projection Loss.

    Modification over vanilla Tree_LoRA:
        - After computing the Tree-guided similarity regulariser (same as before),
          we additionally project the current gradient onto the orthogonal complement
          of the subspace spanned by all previous tasks' accumulated gradients and
          penalise the residual that lies *inside* that subspace:

              L_orth = lambda_orth * sum_{k < t}  || P_k g_t ||^2 / K

          where P_k = v_k v_k^T / ||v_k||^2  is the rank-1 projector for task k's
          mean gradient vector v_k (one projector per LoRA depth layer).

        - `lambda_orth` is controlled by --reg_orth (default 0.1).
          Setting --reg_orth 0 recovers vanilla Tree_LoRA exactly.

    Visualisation:
        - Every `vis_interval` steps we log per-layer cosine similarity between
          g_t and each previous g_k, as well as the orthogonal residual norm.
          These are stored in self.vis_log (list of dicts) so the caller /
          a downstream script can plot them with matplotlib.
    """

    def __init__(self,
                 model, tokenizer, optimizer,
                 train_task_list, eval_task_list, test_task_list, args,
                 lamda_1=0.5, lamda_2=0):
        super().__init__(model, tokenizer, optimizer,
                         train_task_list, eval_task_list, test_task_list, args)

        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.tiktok = TIKTOK(args)

        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

        num_task = len(self.train_task_list)
        args.num_tasks = num_task

        # --reg_orth may not exist in old arg namespaces – fall back to 0.1
        self.lambda_orth = getattr(args, 'reg_orth', 0.1)
        # How often (in steps) to record visualisation stats
        self.vis_interval = getattr(args, 'vis_interval', 50)

        self.kd_lora_tree = KD_LoRA_Tree_Ortho(args)

        # Visualisation log – list of dicts, one entry per vis_interval step
        # Keys: task_id, epoch, step, cos_sim_per_layer (list[list[float]]),
        #       orth_norm_per_layer (list[list[float]]), orth_loss (float)
        self.vis_log = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _collect_loranew_grads(self):
        """Return a list of loranew_A parameter tensors (no copy)."""
        grads = []
        for name_, param_ in self.model.named_parameters():
            if "loranew_A" in name_:
                grads.append(param_)
        return grads

    def _stack_grad(self, grad_list):
        """Stack a list of loranew_A params into (lora_depth, flat_dim)."""
        return torch.stack(
            [g.reshape(-1) for g in grad_list], dim=0
        )  # (lora_depth, D)

    # ------------------------------------------------------------------
    # orthogonal projection loss
    # ------------------------------------------------------------------

    def _orthogonal_projection_loss(self, g_current, task_id):
        """
        Compute orthogonal projection loss.

        For each previous task k and each LoRA depth layer d:
            v_k_d  = all_accumulate_grads[k][d]          (D,)
            P_k_d  = outer(v_k_d, v_k_d) / ||v_k_d||^2  rank-1 projector
            proj   = <g_current[d], v_k_d> / ||v_k_d||^2 * v_k_d
            cost_d = ||proj||^2  =  <g_current[d], v_k_d>^2 / ||v_k_d||^2

        Total loss = lambda_orth / (K * lora_depth)
                     * sum_k sum_d  <g_t_d, v_k_d>^2 / (||v_k_d||^2 + eps)

        This is differentiable w.r.t. g_current (which contains the
        loranew_A parameters) and can be back-propped normally.

        Returns
        -------
        orth_loss : scalar Tensor  (0 if no previous tasks)
        per_layer_orth_norms : list[list[float]]  [prev_k][depth_d]
        """
        eps = 1e-8
        lora_depth = g_current.shape[0]  # number of LoRA layers tracked
        orth_loss = torch.tensor(0.0, device=self.device)
        per_layer_orth_norms = []

        prev_grads = self.kd_lora_tree.all_accumulate_grads[:task_id]
        K = sum(1 for g in prev_grads if g is not None)
        if K == 0:
            return orth_loss, per_layer_orth_norms

        for k, v_k in enumerate(prev_grads):
            if v_k is None:
                continue
            # v_k: (lora_depth, D)  stored on cpu – move to device lazily
            v_k_dev = v_k.to(self.device, non_blocking=True)
            norms_this_k = []

            for d in range(lora_depth):
                g_d = g_current[d]       # (D,)  – differentiable
                v_d = v_k_dev[d]         # (D,)  – detached reference grad

                v_norm_sq = torch.dot(v_d, v_d) + eps
                dot_val   = torch.dot(g_d, v_d.detach())  # detach reference
                proj_norm_sq = dot_val ** 2 / v_norm_sq   # scalar, differentiable

                orth_loss = orth_loss + proj_norm_sq
                norms_this_k.append(proj_norm_sq.item() ** 0.5)

            per_layer_orth_norms.append(norms_this_k)

        orth_loss = self.lambda_orth * orth_loss / (K * lora_depth)
        return orth_loss, per_layer_orth_norms

    # ------------------------------------------------------------------
    # visualisation helper
    # ------------------------------------------------------------------

    def _record_vis(self, task_id, epoch, step, g_current):
        """
        Compute and store cosine similarities and orthogonal norms between
        g_current and every previous task's accumulated gradient,
        per LoRA layer.
        """
        lora_depth = g_current.shape[0]
        eps = 1e-8
        cos_records   = []   # [prev_k][depth_d]
        orth_records  = []   # [prev_k][depth_d]

        prev_grads = self.kd_lora_tree.all_accumulate_grads[:task_id]
        for k, v_k in enumerate(prev_grads):
            if v_k is None:
                continue
            v_k_dev = v_k.to(self.device, non_blocking=True).detach()
            cos_k  = []
            orth_k = []
            for d in range(lora_depth):
                g_d = g_current[d].detach()
                v_d = v_k_dev[d]

                cos_val = (
                    torch.dot(g_d, v_d)
                    / (torch.norm(g_d) * torch.norm(v_d) + eps)
                ).item()

                v_norm_sq = torch.dot(v_d, v_d) + eps
                proj_norm = (torch.dot(g_d, v_d) ** 2 / v_norm_sq).item() ** 0.5

                cos_k.append(cos_val)
                orth_k.append(proj_norm)
            cos_records.append(cos_k)
            orth_records.append(orth_k)

        entry = dict(
            task_id=task_id,
            epoch=epoch,
            step=step,
            cos_sim_per_layer=cos_records,
            orth_norm_per_layer=orth_records,
        )
        self.vis_log.append(entry)

    # ------------------------------------------------------------------
    # main training loop
    # ------------------------------------------------------------------

    def train_one_task(self, task, task_id, epochs):
        num_task        = len(self.train_task_list)
        train_dataloader = self.train_task_list[task]
        eval_dataloader  = self.eval_task_list[task]

        total_steps          = epochs * len(train_dataloader)
        train_dataloader_len = len(train_dataloader)
        progress_bar = tqdm(
            total=total_steps, leave=True,
            disable=(self.args.global_rank != 0)
        )

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Beginning of Epoch {epoch + 1}/{epochs}, "
                f"Total Micro Batches {train_dataloader_len}",
                self.args.global_rank
            )
            self.model.train()
            self.tiktok.print_time(self.args.global_rank)

            self.kd_lora_tree.new_epoch_init(train_dataloader_len)
            tmp_rounds = -1

            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1

                if self.args.reg > 0:
                    self.kd_lora_tree.step()

                del batch['sources']
                batch   = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss    = outputs.loss

                # ---- collect loranew_A gradients (as parameters, differentiable) ----
                if self.args.reg > 0:
                    self.tiktok.tik()
                    grad_params = self._collect_loranew_grads()
                    _grad_current = self._stack_grad(grad_params)   # (lora_depth, D)
                    self.tiktok.tok(f"Calculate_Grad_@Task{task_id} Epoch{epoch}")

                    self.tiktok.tik()
                    self.kd_lora_tree.insert_grad(_grad_current)
                    self.tiktok.tok(f"Insert_Grad_@Task{task_id} Epoch{epoch}")

                    # ---- Tree-guided similarity regulariser (unchanged) ----
                    if task_id > 0:
                        self.tiktok.tik()
                        prev_id_matrix = self.kd_lora_tree.tree_search(
                            task_id, device=self.device
                        )
                        self.tiktok.tok(
                            f"Calculate_Tree_Search_@Task{task_id} Epoch{epoch}"
                        )

                        self.tiktok.tik()
                        reg_loss = self.kd_lora_tree.get_loss(
                            _grad_current, loss, task_id, prev_id_matrix
                        )
                        loss = loss - reg_loss
                        self.tiktok.tok(
                            f"Calculate_Tree_Reg_@Task{task_id} Epoch{epoch}"
                        )

                        # ---- Orthogonal Projection Loss (new) ----
                        self.tiktok.tik()
                        orth_loss, orth_norms = self._orthogonal_projection_loss(
                            _grad_current, task_id
                        )
                        loss = loss + orth_loss
                        self.tiktok.tok(
                            f"Calculate_Orth_Loss_@Task{task_id} Epoch{epoch}"
                        )

                        # ---- periodic logging ----
                        if tmp_rounds % 100 == 0:
                            print_rank_0(
                                f"\033[34m(Ortho) Sim: {self.kd_lora_tree.sim};\033[0m",
                                self.args.global_rank
                            )
                            print_rank_0(
                                f"\033[34m(Ortho) Selected Nums: "
                                f"{self.kd_lora_tree.num_of_selected[:task_id]};\033[0m",
                                self.args.global_rank
                            )
                            print_rank_0(
                                f"\033[34m(Ortho) Prev_id_matrix: {prev_id_matrix};\033[0m",
                                self.args.global_rank
                            )
                            print_rank_0(
                                f"\033[34mReg Loss: {reg_loss:.4f} | "
                                f"Orth Loss: {orth_loss.item():.4f}\033[0m",
                                self.args.global_rank
                            )

                        # ---- visualisation recording ----
                        if tmp_rounds % self.vis_interval == 0:
                            self._record_vis(task_id, epoch, step, _grad_current)
                            # attach orth_loss scalar to last record
                            self.vis_log[-1]['orth_loss'] = orth_loss.item()

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = (
                        f"Epoch {epoch + 1}, Step {step}, "
                        f"Loss: {loss.item():.4f}"
                    )
                    progress_bar.set_description(description, refresh=False)

                self.tiktok.tik()
                self.model.backward(loss)
                self.model.step()
                self.tiktok.tok('backward time')

                if self.args.global_rank == 0 and tmp_rounds % 30 == 0:
                    self.tiktok.print_time()

        # ---- save ----
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_id))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(
                f'Successfully saving the final model to {peft_model_id}',
                self.args.global_rank
            )

            # save visualisation log as JSON
            vis_path = os.path.join(
                self.args.output_dir,
                f'vis_log_task{task_id}.json'
            )
            with open(vis_path, 'w') as f:
                json.dump(self.vis_log, f, indent=2)
            print_rank_0(
                f'Visualisation log saved to {vis_path}',
                self.args.global_rank
            )

        if self.args.reg > 0:
            self.kd_lora_tree.end_task(task_id=task_id)

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
            print_rank_0(
                f'Successfully saving the final model to {peft_model_id}',
                self.args.global_rank
            )