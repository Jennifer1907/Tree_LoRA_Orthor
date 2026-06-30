import json
import os

import torch
from tqdm import tqdm

from model.base_model import CL_Base_Model
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device


# ---------------------------------------------------------------------------
# LoRA Baseline
# ---------------------------------------------------------------------------
# Ablation variant: vanilla LoRA fine-tuning with NO regularisation.
#
# Loss at every step:
#     total_loss = loss_ce
#
# Purpose: lower-bound baseline.  Measures catastrophic forgetting without
# any continual-learning regulariser.
#
# Removed vs Tree_LoRA_Ortho:
#   - KD_LoRA_Tree  (no tree_search, no insert_grad, no get_loss)
#   - Orthogonal Projection Loss (_orth_loss_stop_grad)
#   - autograd.grad  call  (not needed without reg)
#   - gradient accumulation  (_accum_grad / _accum_steps)
#   - visualisation log  (nothing useful to log without grad signals)
# ---------------------------------------------------------------------------


class LoRA_Baseline(CL_Base_Model):
    """
    Vanilla LoRA continual learning – no KD-tree, no OPL.

    total_loss = loss_ce

    Args accepted (ignored silently so arg-parsing stays identical):
        args.reg       – ignored (set to 0 internally)
        args.reg_orth  – ignored
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

        # Force-disable both regularisers so any downstream code that reads
        # args.reg / args.reg_orth still gets a consistent value.
        self.args.reg      = 0
        self.args.reg_orth = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_task(self, task, task_id, epochs):
        train_dataloader     = self.train_task_list[task]
        train_dataloader_len = len(train_dataloader)
        total_steps          = epochs * train_dataloader_len
        progress_bar = tqdm(total=total_steps, leave=True,
                            disable=(self.args.global_rank != 0))

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, "
                f"Total Micro Batches {train_dataloader_len}",
                self.args.global_rank)

            self.model.train()
            self.tiktok.print_time(self.args.global_rank)

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, self.device)

                # ── Forward ──────────────────────────────────────────────
                outputs    = self.model(**batch, use_cache=False)
                loss_ce    = outputs.loss
                total_loss = loss_ce          # ← only CE loss

                # ── Backward ─────────────────────────────────────────────
                self.tiktok.tik()
                self.model.backward(total_loss)
                self.model.step()
                self.tiktok.tok('backward_step')

                # ── Progress bar ──────────────────────────────────────────
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Epoch {epoch+1}, Step {step}, "
                        f"CE: {loss_ce.item():.4f}",
                        refresh=False)

                if self.args.global_rank == 0 and step % 30 == 0:
                    self.tiktok.print_time()

        # ── Save checkpoint ───────────────────────────────────────────────
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_id))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Saved model → {peft_model_id}', self.args.global_rank)

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