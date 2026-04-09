import json
import os
import math
import torch
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.kd_lora_tree import KD_LoRA_Tree
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_loranew_A_params(model):
    """Return (name, param) pairs for every loranew_A parameter, stable order."""
    return [(n, p) for n, p in model.named_parameters() if "loranew_A" in n]


def _collect_real_grads(params_and_names):
    """
    After backward(), gather .grad buffers for each loranew_A param.
    Returns a (D, flat_dim) float32 tensor on CPU, or None if grads missing.
    """
    grads = []
    for _, p in params_and_names:
        if p.grad is None:
            return None
        grads.append(p.grad.detach().float().reshape(-1))
    return torch.stack(grads, dim=0)   # (D, flat_dim)


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
        create_graph=False,  # No 2nd-order; too expensive for bfloat16+DS
        allow_unused=True,
    )
    g_list = []
    for g, (_, p) in zip(grads, params_and_names):
        if g is None:
            g_list.append(torch.zeros(p.numel(), dtype=torch.float32))
        else:
            g_list.append(g.float().reshape(-1))
    return torch.stack(g_list, dim=0)  # (D, flat_dim)


# ---------------------------------------------------------------------------
# Orthogonal projection loss  (stop-gradient / GradOrth trick)
# ---------------------------------------------------------------------------

def _orth_loss_stop_grad(params_and_names, g_t_detached, all_prev_grads, reg_orth):
    """
    L_orth = (reg_orth / (K*D)) * Σ_k Σ_d  α_{k,d} * <param_d_flat, v_{k,d}>

    where  α_{k,d} = <g_t_d (detached), v_{k,d}> / ||v_{k,d}||²   (scalar, detached)

    The gradient w.r.t. param_d is  Σ_k α_{k,d} * v_{k,d},
    which pushes parameters away from the previous-task gradient subspace.

    Args:
        params_and_names : list[(name, param)] for loranew_A layers
        g_t_detached     : (D, flat_dim) float32 CPU, DETACHED
        all_prev_grads   : (K, D, flat_dim) float32 CPU
        reg_orth         : float

    Returns:
        orth_loss  : differentiable scalar (through params)
        proj_norms : list[float] of length D, max ||α·v|| per layer
    """
    K, D, _ = all_prev_grads.shape
    orth_loss = None
    proj_norms = []

    for d, (_, param) in enumerate(params_and_names):
        # param lives on CUDA; work in float32 for numerical stability
        param_flat = param.reshape(-1).float()   # differentiable, on CUDA
        g_d = g_t_detached[d]                    # CPU, detached

        layer_max_proj = 0.0
        for k in range(K):
            v_k_d = all_prev_grads[k, d]                       # CPU float32
            v_norm_sq = torch.dot(v_k_d, v_k_d).item()
            if v_norm_sq < 1e-12:
                continue

            # α: detached scalar — direction of alignment
            alpha = (torch.dot(g_d, v_k_d) / v_norm_sq).detach()  # scalar CPU

            # Differentiable term; gradient = α * v_k_d
            v_k_d_cuda = v_k_d.to(param_flat.device)
            term = alpha.to(param_flat.device) * torch.dot(param_flat, v_k_d_cuda)

            orth_loss = term if orth_loss is None else orth_loss + term

            # Visualisation: ||α * v||
            proj_norm = (alpha.item() ** 2 * v_norm_sq) ** 0.5
            layer_max_proj = max(layer_max_proj, proj_norm)

        proj_norms.append(layer_max_proj)

    if orth_loss is None:
        return torch.tensor(0.0), proj_norms

    scale = reg_orth / max(K * D, 1)
    return orth_loss * scale, proj_norms


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _cosine_sim_per_layer(g_t, all_prev_grads):
    """
    Cosine similarity between g_t and each previous-task gradient, per layer,
    averaged across tasks.
    Both tensors are float32 CPU.

    Args:
        g_t            : (D, flat_dim)
        all_prev_grads : (K, D, flat_dim)
    Returns:
        list[float] of length D
    """
    K, D, _ = all_prev_grads.shape
    cos_sims = []
    for d in range(D):
        g = g_t[d]
        g_norm = torch.norm(g).item()
        total, count = 0.0, 0
        for k in range(K):
            v = all_prev_grads[k, d]
            v_norm = torch.norm(v).item()
            if g_norm < 1e-12 or v_norm < 1e-12:
                continue
            total += torch.dot(g, v).item() / (g_norm * v_norm)
            count += 1
        cos_sims.append(total / max(count, 1))
    return cos_sims


# ---------------------------------------------------------------------------
# Tree_LoRA_Ortho
# ---------------------------------------------------------------------------

class Tree_LoRA_Ortho(CL_Base_Model):
    """
    Continual-learning trainer combining:
      - KD-tree gradient similarity regulariser  (Tree_LoRA)
      - Orthogonal projection loss               (new)

    Final per-step loss:
        total_loss = loss_ce  -  reg_loss  +  orth_loss

    Extra args (with defaults):
        args.reg          - weight for KD-tree reg     (default 0.5)
        args.reg_orth     - weight for orth loss       (default 0.1)
        args.vis_interval - steps between vis entries  (default 50)

    Execution order inside each step (critical for DeepSpeed + autograd):
        1. forward()                              → loss_ce
        2. autograd.grad(loss_ce, retain_graph)   → g_t  (pure gradients)
        3. kd_tree.insert_grad / tree_search      → prev_id_matrix
        4. kd_tree.get_loss                       → reg_loss
        5. _orth_loss_stop_grad                   → orth_loss  (through params)
        6. total_loss = loss_ce - reg_loss + orth_loss
        7. model.backward(total_loss)             → fills .grad buffers
        8. collect .grad buffers                  → accumulate for KD-tree
        9. model.step()
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

        num_task        = len(self.train_task_list)
        args.num_tasks  = num_task
        self.kd_lora_tree = KD_LoRA_Tree(args)

        self._vis_log   = []   # reset each task
        self._accum_grad  = None
        self._accum_steps = 0

    # ------------------------------------------------------------------
    # Gradient accumulation (real .grad, not param values)
    # ------------------------------------------------------------------

    def _reset_grad_accum(self):
        self._accum_grad  = None
        self._accum_steps = 0

    def _accumulate_real_grad(self, params_and_names):
        """Add the current .grad buffers to the running sum (CPU, float32)."""
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
        """Store the per-task mean gradient into the KD-tree."""
        if self._accum_steps == 0 or self._accum_grad is None:
            print_rank_0(
                f"[WARNING] No gradients accumulated for task {task_id}.",
                self.args.global_rank)
            return
        mean_grad = self._accum_grad / self._accum_steps
        # kd_lora_tree.current_grad is read by end_task()
        self.kd_lora_tree.current_grad = mean_grad.to(self.device)

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

            self.kd_lora_tree.new_epoch_init(train_dataloader_len)
            self._reset_grad_accum()

            # Collect loranew_A params once — order is stable within an epoch
            params_and_names = _collect_loranew_A_params(self.model)

            tmp_rounds = -1

            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1

                if self.args.reg > 0:
                    self.kd_lora_tree.step()

                del batch['sources']
                batch = to_device(batch, self.device)

                # ---- Step 1: Forward ----------------------------------------
                outputs = self.model(**batch, use_cache=False)
                loss_ce = outputs.loss

                reg_loss  = torch.tensor(0.0, device=self.device)
                orth_loss = torch.tensor(0.0, device=self.device)
                proj_norms = []
                prev_id_matrix = None
                all_prev_grads_cpu = None

                if self.args.reg > 0:
                    # ---- Step 2: True gradients via autograd ----------------
                    #              MUST happen before model.backward()
                    self.tiktok.tik()
                    g_t = _stack_grad_via_autograd(loss_ce, params_and_names)
                    # g_t: (D, flat_dim) float32, no grad tracking
                    self.tiktok.tok("autograd_grad @T{} E{}".format(task_id, epoch))

                    # Send to device for KD-tree operations
                    g_t_device = g_t.to(self.device)

                    # Insert into per-epoch accumulator (mirrors Tree_LoRA)
                    self.kd_lora_tree.insert_grad(g_t_device)

                    if task_id > 0:
                        # ---- Step 3: KD-tree search -------------------------
                        self.tiktok.tik()
                        prev_id_matrix = self.kd_lora_tree.tree_search(
                            task_id, device=self.device)
                        self.tiktok.tok("tree_search @T{} E{}".format(task_id, epoch))

                        # ---- Step 4: KD-tree reg loss -----------------------
                        self.tiktok.tik()
                        reg_loss = self.kd_lora_tree.get_loss(
                            g_t_device, loss_ce, task_id, prev_id_matrix)
                        self.tiktok.tok("kd_reg_loss @T{} E{}".format(task_id, epoch))

                        # ---- Step 5: Orthogonal projection loss -------------
                        if self.reg_orth > 0:
                            self.tiktok.tik()
                            all_prev_grads = self.kd_lora_tree.all_grad_device
                            # all_prev_grads: (K, D, flat_dim) on device

                            if all_prev_grads is not None:
                                all_prev_grads_cpu = all_prev_grads.float().cpu()
                                g_t_det_cpu = g_t.detach().cpu()  # already no grad

                                orth_loss, proj_norms = _orth_loss_stop_grad(
                                    params_and_names,
                                    g_t_det_cpu,
                                    all_prev_grads_cpu,
                                    self.reg_orth,
                                )
                                orth_loss = orth_loss.to(self.device)
                            else:
                                proj_norms = [0.0] * len(params_and_names)

                            self.tiktok.tok("orth_loss @T{} E{}".format(task_id, epoch))

                        # Periodic console log
                        if tmp_rounds % 100 == 0:
                            print_rank_0(
                                "\033[34m[Ortho] reg={:.4f}  orth={:.4f}  sim={}\033[0m".format(
                                    reg_loss.item(),
                                    orth_loss.item(),
                                    self.kd_lora_tree.sim,
                                ),
                                self.args.global_rank)
                            print_rank_0(
                                "\033[34m[Ortho] num_selected={}\033[0m".format(
                                    self.kd_lora_tree.num_of_selected[:task_id]),
                                self.args.global_rank)
                            print_rank_0(
                                "\033[34m[Ortho] prev_id_matrix={}\033[0m".format(
                                    prev_id_matrix),
                                self.args.global_rank)

                # ---- Step 6: Combine losses ---------------------------------
                total_loss = loss_ce - reg_loss + orth_loss

                # ---- Step 7: Backward (DeepSpeed) ---------------------------
                self.tiktok.tik()
                self.model.backward(total_loss)
                self.model.step()
                self.tiktok.tok('backward_step')

                # ---- Step 8: Collect real .grad buffers ---------------------
                #              These are used for KD-tree storage (end_task)
                if self.args.reg > 0:
                    self._accumulate_real_grad(params_and_names)

                # ---- Step 9: Visualisation log (rank-0 only) ----------------
                if (self.args.global_rank == 0
                        and task_id > 0
                        and tmp_rounds % self.vis_interval == 0):
                    if all_prev_grads_cpu is not None:
                        cos_sims = _cosine_sim_per_layer(
                            g_t.detach().cpu(), all_prev_grads_cpu)
                    else:
                        cos_sims = [0.0] * len(params_and_names)

                    self._vis_log.append({
                        "step":               step,
                        "epoch":              epoch,
                        "cos_sim_per_layer":  cos_sims,
                        "orth_norm_per_layer": proj_norms,
                        "orth_loss":          orth_loss.item()
                                              if torch.is_tensor(orth_loss)
                                              else 0.0,
                    })

                # Progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Epoch {epoch+1}, Step {step}, "
                        f"CE: {loss_ce.item():.4f}, "
                        f"Orth: {orth_loss.item() if torch.is_tensor(orth_loss) else 0.0:.4f}",
                        refresh=False)

                if self.args.global_rank == 0 and tmp_rounds % 30 == 0:
                    self.tiktok.print_time()

        # ---- End of all epochs for this task --------------------------------

        # Finalise task gradient for KD-tree (uses real .grad, not params)
        if self.args.reg > 0:
            self._finalize_task_grad(task_id)

        # ---- Save checkpoint ------------------------------------------------
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_id))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Saved model → {peft_model_id}', self.args.global_rank)

            # Visualisation log
            vis_path = os.path.join(self.args.output_dir,
                                    f"vis_log_task{task_id}.json")
            with open(vis_path, 'w') as f:
                json.dump(self._vis_log, f, indent=2)
            print_rank_0(f'Saved vis log → {vis_path}', self.args.global_rank)

        # Update KD-tree with finalised gradient (builds/rebuilds the tree)
        if self.args.reg > 0:
            self.kd_lora_tree.end_task(task_id=task_id)

    # ------------------------------------------------------------------
    # save_model  (called externally; mirrors Tree_LoRA)
    # ------------------------------------------------------------------

    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(round))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)

            # Patch adapter_config for O_LoRA compatibility
            adapter_config_path = os.path.join(peft_model_id, 'adapter_config.json')
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            adapter_config['r_sum'] = 0
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f)

            print_rank_0(f'Saved model → {peft_model_id}', self.args.global_rank)