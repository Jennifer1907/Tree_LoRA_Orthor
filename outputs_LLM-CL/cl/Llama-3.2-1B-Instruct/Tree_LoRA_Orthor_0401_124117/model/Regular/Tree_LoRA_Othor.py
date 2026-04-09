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
from utils.kd_lora_tree import KD_LoRA_Tree
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


class Tree_LoRA(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 lamda_1=0.5, lamda_2=0
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        '''
        orthological to previous adapters
        '''
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
        self.kd_lora_tree = KD_LoRA_Tree(args)

    def train_one_task(self, task, task_id, epochs):
        num_task = len(self.train_task_list)
        train_dataloader = self.train_task_list[task]
        eval_dataloader  = self.eval_task_list[task]

        #### TRAIN ####
        total_steps          = epochs * len(train_dataloader)
        train_dataloader_len = len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Beginning of Epoch {epoch + 1}/{epochs}, Total Micro Batches {train_dataloader_len}",
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

                del batch['sources']
                batch   = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss    = outputs.loss

                if self.args.reg > 0:
                    self.tiktok.tik()
                    # Collect loranew_A parameters as the current gradient proxy
                    _grad_current = []
                    for name_, param_ in self.model.named_parameters():
                        if "loranew_A" in name_:
                            _grad_current.append(param_)   # (r, dim)
                    self.tiktok.tok("Calculate_Grad_@Task{} Epoch{}".format(task_id, epoch))

                    self.tiktok.tik()
                    # Stack into (lora_depth, dim * rank)
                    _grad_current = torch.stack(
                        [_grad_current[i].reshape(-1) for i in range(len(_grad_current))], dim=0
                    )
                    self.kd_lora_tree.insert_grad(_grad_current)
                    self.tiktok.tok("Split_Grad_@Task{} Epoch{}".format(task_id, epoch))

                    if task_id > 0:
                        self.tiktok.tik()
                        prev_id_matrix = self.kd_lora_tree.tree_search(task_id, device=self.device)
                        self.tiktok.tok("Calculate_Tree_Search_@Task{} Epoch{}".format(task_id, epoch))

                        self.tiktok.tik()
                        # get_loss now returns sim_loss + lamda_opl * opl_loss
                        reg_loss = self.kd_lora_tree.get_loss(
                            _grad_current, loss, task_id, prev_id_matrix
                        )
                        loss = loss - reg_loss
                        self.tiktok.tok("Calculate_Tree_Reg_@Task{} Epoch{}".format(task_id, epoch))

                        if tmp_rounds % 100 == 0:
                            print_rank_0(
                                "\033[34m(Normal Process) Sim: {};\033[0m".format(
                                    self.kd_lora_tree.sim
                                ),
                                self.args.global_rank,
                            )
                            print_rank_0(
                                "\033[34m(Normal Process) Selected Nums: {};\033[0m".format(
                                    self.kd_lora_tree.num_of_selected[:task_id]
                                ),
                                self.args.global_rank,
                            )
                            print_rank_0(
                                "\033[34m(Normal Process) Prev_id_matrix: {};\033[0m".format(
                                    prev_id_matrix
                                ),
                                self.args.global_rank,
                            )
                            print_rank_0(
                                "\033[34mReg Loss: {:.4f}\033[0m".format(reg_loss),
                                self.args.global_rank,
                            )
                            # Log OPL basis rank info when OPL is active
                            if getattr(self.args, 'lamda_opl', 0.0) > 0:
                                basis_info = self.kd_lora_tree.opl_bases[task_id - 1]
                                if basis_info is not None:
                                    k_sizes = [
                                        b.shape[1] if b is not None else 0
                                        for b in basis_info
                                    ]
                                    print_rank_0(
                                        "\033[34mOPL basis ranks per depth: {};\033[0m".format(
                                            k_sizes
                                        ),
                                        self.args.global_rank,
                                    )

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.tiktok.tik()
                self.model.backward(loss)
                self.model.step()
                self.tiktok.tok('backward time')

                if self.args.global_rank == 0:
                    if tmp_rounds % 30 == 0:
                        self.tiktok.print_time()

        #### SAVE ####
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_id))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(
                f'Successfully saving the final model to {peft_model_id}',
                self.args.global_rank,
            )

        if self.args.reg > 0:
            # After each task: freeze accumulated gradient and (re)build OPL bases
            self.kd_lora_tree.end_task(task_id=task_id)

    def save_model(self, round):
        #### SAVE ####
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(round))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)

            adapter_config_path = os.path.join(peft_model_id, 'adapter_config.json')
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            # Reset r_sum to 0 for O_LoRA compatibility
            adapter_config['r_sum'] = 0
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f)

            print_rank_0(
                f'Successfully saving the final model to {peft_model_id}',
                self.args.global_rank,
            )