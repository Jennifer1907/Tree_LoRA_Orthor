"""
visualize_gradient_heatmap.py
─────────────────────────────
Reads vis_log_task{N}.json files produced by Tree_LoRA_Ortho and draws
heatmaps matching Fig. 3 of the paper:
  - Left  : Basis Vector Usage   (how often each basis / rank is selected)
  - Right : Gradient Soft Mask   (projection norm ||α·v|| per layer × rank)

Usage
-----
python visualize_gradient_heatmap.py \
    --log_dir  /path/to/output_dir \
    --task_ids 1 2 3 4 5 6 7 \
    --layers   0 6 11 \
    --out_dir  ./figs

The script reads vis_log_task{N}.json for each task_id.
Each JSON is a list of dicts with keys:
    step, epoch, cos_sim_per_layer, orth_norm_per_layer, orth_loss

Additional keys written by the patched trainer (see below):
    basis_usage_per_layer   : list[list[float]]  shape (D, R)
    grad_mask_per_layer     : list[list[float]]  shape (D, R)

──────────────────────────────────────────────────────────────────────────────
PATCH for tree_lora_ortho.py
──────────────────────────────────────────────────────────────────────────────
Add the two functions below and call them inside the vis-log block.

    def _basis_usage(kd_lora_tree, task_id, D):
        \"\"\"
        Per-layer basis-vector usage: num_of_selected normalised to [0,1].
        Returns (D, R) list where R = number of previous tasks (= rank proxy).
        \"\"\"
        if kd_lora_tree.num_of_selected is None:
            return [[0.0]] * D
        sel = kd_lora_tree.num_of_selected[:task_id, :].float()  # (K, D)
        sel = sel.T  # (D, K)
        max_val = sel.max()
        if max_val < 1e-9:
            return sel.cpu().tolist()
        return (sel / max_val).cpu().tolist()   # (D, K)

    def _grad_soft_mask(g_t_cpu, all_prev_grads_cpu):
        \"\"\"
        Projection norm per (layer, prev_task): |<g_t_d, v_k_d>|² / ||v_k_d||²
        Returns (D, K) list.
        \"\"\"
        if all_prev_grads_cpu is None:
            return []
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
                    alpha_sq = (torch.dot(g_t_cpu[d], v) ** 2 / v_norm_sq).item()
                    row.append(float(alpha_sq) ** 0.5)  # ||proj||
            mask.append(row)
        return mask   # (D, K)

Then inside the vis-log block in train_one_task, append:

    entry["basis_usage_per_layer"] = _basis_usage(
        self.kd_lora_tree, task_id, len(params_and_names))
    entry["grad_mask_per_layer"]   = _grad_soft_mask(
        g_t.detach().cpu(), all_prev_grads_cpu)

──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ─── colour maps matching the paper ──────────────────────────────────────────
CMAP_BASIS = "viridis"       # cool–green–yellow (basis usage)
CMAP_MASK  = "plasma"        # purple–magenta–yellow (soft mask)


def load_log(log_dir: str, task_id: int) -> list[dict]:
    path = Path(log_dir) / f"vis_log_task{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def _pad_rows(rows: list[list[float]], ncols: int) -> np.ndarray:
    """Pad or trim each row to exactly ncols, return (nrows, ncols) float32."""
    out = np.zeros((len(rows), ncols), dtype=np.float32)
    for i, r in enumerate(rows):
        n = min(len(r), ncols)
        out[i, :n] = r[:n]
    return out


def build_matrices(log: list[dict], target_layers: list[int], n_ranks: int = 18):
    """
    Returns two dicts keyed by layer_idx:
        basis[layer]  : (n_epochs, n_ranks)  – mean basis usage per epoch
        mask[layer]   : (n_epochs, n_ranks)  – mean gradient soft mask per epoch
    n_ranks: how many rank/task columns to show (pad with 0 if fewer available)
    """
    # Group entries by epoch
    epochs: dict[int, list[dict]] = {}
    for entry in log:
        e = entry.get("epoch", 0)
        epochs.setdefault(e, []).append(entry)

    sorted_epochs = sorted(epochs.keys())
    n_epochs = len(sorted_epochs)

    basis_mat = {L: np.zeros((n_epochs, n_ranks), np.float32) for L in target_layers}
    mask_mat  = {L: np.zeros((n_epochs, n_ranks), np.float32) for L in target_layers}

    for ei, e in enumerate(sorted_epochs):
        entries = epochs[e]

        # ── basis usage ──────────────────────────────────────────────────────
        # basis_usage_per_layer: list of D lists (D layers, each list = K values)
        all_basis = [
            ent.get("basis_usage_per_layer", [])
            for ent in entries
            if ent.get("basis_usage_per_layer")
        ]
        if all_basis:
            # average over steps within the epoch
            # each element: (D, K) ragged; use the last step as representative
            basis_snapshot = all_basis[-1]  # (D, K) as list-of-lists
            for L in target_layers:
                if L < len(basis_snapshot):
                    row = basis_snapshot[L]
                    n = min(len(row), n_ranks)
                    basis_mat[L][ei, :n] = row[:n]
        else:
            # Fallback: use cos_sim_per_layer as proxy (scalar per layer)
            for ent in entries:
                cos = ent.get("cos_sim_per_layer", [])
                for L in target_layers:
                    if L < len(cos):
                        # broadcast a single scalar across all "ranks"
                        basis_mat[L][ei, :] += abs(cos[L]) / len(entries)

        # ── gradient soft mask ───────────────────────────────────────────────
        all_mask = [
            ent.get("grad_mask_per_layer", [])
            for ent in entries
            if ent.get("grad_mask_per_layer")
        ]
        if all_mask:
            mask_snapshot = all_mask[-1]
            for L in target_layers:
                if L < len(mask_snapshot):
                    row = mask_snapshot[L]
                    n = min(len(row), n_ranks)
                    mask_mat[L][ei, :n] = row[:n]
        else:
            # Fallback: use orth_norm_per_layer
            for ent in entries:
                orth = ent.get("orth_norm_per_layer", [])
                for L in target_layers:
                    if L < len(orth):
                        mask_mat[L][ei, :] += orth[L] / len(entries)

    return basis_mat, mask_mat, sorted_epochs


def draw_fig(
    basis_mats: dict,      # task_id -> {layer -> (E, R)}
    mask_mats: dict,
    task_ids: list[int],
    target_layers: list[int],
    n_ranks: int,
    out_path: str,
):
    """
    Layout mirrors Fig. 3:
      rows    = target_layers  (Layer 0 / Layer 6 / Layer 11)
      columns = task_ids × 2   (Basis | Mask per task)
    For a single task the two column groups collapse into one pair.
    """
    n_layers = len(target_layers)
    n_tasks  = len(task_ids)

    # Figure layout: 2 column groups (Basis | Mask), each group has n_tasks sub-columns
    total_cols = n_tasks * 2
    fig_w = max(8, total_cols * 1.6)
    fig_h = max(4, n_layers * 1.8)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # Top-level title columns
    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig, wspace=0.18,
        left=0.08, right=0.96, top=0.88, bottom=0.10,
    )

    axes_basis = []  # [task][layer]
    axes_mask  = []

    for gi, (mats, cmap, col_label, axes_list) in enumerate([
        (basis_mats, CMAP_BASIS, "Basis Vector Usage",  axes_basis),
        (mask_mats,  CMAP_MASK,  "Gradient Soft Mask",  axes_mask),
    ]):
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            n_layers, n_tasks,
            subplot_spec=gs_outer[gi],
            hspace=0.08, wspace=0.06,
        )
        fig.text(
            0.25 + gi * 0.5, 0.94,
            col_label, ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

        task_axes = []
        for ti, task_id in enumerate(task_ids):
            layer_axes = []
            for li, L in enumerate(target_layers):
                ax = fig.add_subplot(gs_inner[li, ti])

                mat = mats.get(task_id, {}).get(L)
                if mat is None or mat.sum() == 0:
                    ax.set_visible(False)
                    layer_axes.append(ax)
                    continue

                # normalise per subplot for visual clarity
                vmax = np.percentile(mat, 99) or 1.0
                im = ax.imshow(
                    mat, aspect="auto", origin="lower",
                    cmap=cmap, vmin=0, vmax=vmax,
                    interpolation="nearest",
                )

                # Axis decoration
                n_e = mat.shape[0]
                xtick_step = max(1, n_ranks // 6)
                ytick_step = max(1, n_e // 4)

                ax.set_xticks(range(0, n_ranks, xtick_step))
                ax.set_xticklabels(
                    [f"T{i+1}" for i in range(0, n_ranks, xtick_step)],
                    fontsize=5,
                )
                ax.set_yticks(range(0, n_e, ytick_step))
                ax.set_yticklabels(
                    [str(i) for i in range(0, n_e, ytick_step)],
                    fontsize=5,
                )
                ax.tick_params(length=2, pad=1)

                # Row label: "Layer X Rank" on the very first column of the
                # left group only
                if gi == 0 and ti == 0:
                    ax.set_ylabel(
                        f"Layer {L}\nRank",
                        fontsize=6, labelpad=3,
                        rotation=0, ha="right", va="center",
                    )
                    ax.yaxis.set_label_coords(-0.28, 0.5)
                else:
                    ax.set_yticks([])

                # Column header: task id on the first row only
                if li == 0:
                    ax.set_title(
                        f"T{task_id}", fontsize=6, pad=3, fontweight="normal"
                    )

                # X label on bottom row only
                if li == n_layers - 1:
                    ax.set_xlabel("Training Epochs", fontsize=5, labelpad=2)
                else:
                    ax.set_xticks([])

                # Thin colour bar on rightmost column
                if ti == n_tasks - 1:
                    cb = plt.colorbar(im, ax=ax, fraction=0.12, pad=0.04)
                    cb.ax.tick_params(labelsize=4)

                layer_axes.append(ax)
            task_axes.append(layer_axes)
        axes_list.append(task_axes)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",  required=True,
                        help="Directory containing vis_log_task{N}.json files")
    parser.add_argument("--task_ids", nargs="+", type=int, default=[1, 2, 3],
                        help="Which tasks to plot")
    parser.add_argument("--layers",   nargs="+", type=int, default=[0, 6, 11],
                        help="Which LoRA layers to show (0-indexed)")
    parser.add_argument("--n_ranks",  type=int, default=18,
                        help="Number of rank/task columns on the x-axis")
    parser.add_argument("--out_dir",  default="./figs")
    parser.add_argument("--per_task", action="store_true",
                        help="Also save one figure per task")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── load all logs ──────────────────────────────────────────────────────────
    basis_all: dict[int, dict] = {}
    mask_all:  dict[int, dict] = {}

    for task_id in args.task_ids:
        try:
            log = load_log(args.log_dir, task_id)
        except FileNotFoundError as e:
            print(f"[WARN] {e} — skipping task {task_id}")
            continue

        basis_m, mask_m, epochs = build_matrices(log, args.layers, args.n_ranks)
        basis_all[task_id] = basis_m
        mask_all[task_id]  = mask_m
        print(f"Task {task_id}: {len(log)} entries, {len(epochs)} epochs")

        if args.per_task:
            out_path = os.path.join(
                args.out_dir, f"gradient_heatmap_task{task_id}.png")
            draw_fig(
                {task_id: basis_m}, {task_id: mask_m},
                [task_id], args.layers, args.n_ranks, out_path,
            )

    # ── combined figure (all tasks side by side) ──────────────────────────────
    valid_tasks = [t for t in args.task_ids if t in basis_all]
    if valid_tasks:
        out_path = os.path.join(args.out_dir, "gradient_heatmap_all_tasks.png")
        draw_fig(
            basis_all, mask_all,
            valid_tasks, args.layers, args.n_ranks, out_path,
        )


if __name__ == "__main__":
    main()