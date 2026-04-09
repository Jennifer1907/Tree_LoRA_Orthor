"""
visualize_gradient_comparison.py
──────────────────────────────────
Load the vis_log JSON files saved by Tree_LoRA_Ortho and produce:
  1. Per-layer cosine similarity over training steps  (with/without ortho)
  2. Per-layer orthogonal projection norm over steps
  3. Orthogonal loss curve

Usage
-----
python visualize_gradient_comparison.py \
    --log_ortho  outputs/.../vis_log_task1.json \
    [--log_base  outputs_baseline/.../vis_log_task1.json] \
    --out_dir    figures/

If --log_base is omitted only the ortho curves are plotted.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def load_log(path):
    with open(path) as f:
        return json.load(f)


def extract_series(log, prev_task=0):
    """
    Pull time-series from a vis_log list.

    Returns
    -------
    steps      : list[int]
    cos_layers : list of lists  [layer_d][time]
    orth_layers: list of lists  [layer_d][time]
    orth_loss  : list[float]    (scalar per step, may contain None)
    """
    steps      = []
    cos_all    = None
    orth_all   = None
    orth_loss  = []

    for entry in log:
        cos_data  = entry.get('cos_sim_per_layer',  [])
        orth_data = entry.get('orth_norm_per_layer', [])
        ol        = entry.get('orth_loss', None)

        # find the entry for prev_task index
        if prev_task >= len(cos_data):
            continue

        cos_vec  = cos_data[prev_task]   # list[float] over layers
        orth_vec = orth_data[prev_task]

        n_layers = len(cos_vec)
        if cos_all is None:
            cos_all  = [[] for _ in range(n_layers)]
            orth_all = [[] for _ in range(n_layers)]

        steps.append(entry['step'])
        for d in range(n_layers):
            cos_all[d].append(cos_vec[d])
            orth_all[d].append(orth_vec[d])
        orth_loss.append(ol)

    return steps, cos_all or [], orth_all or [], orth_loss


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_cosine_similarity(steps_ortho, cos_ortho,
                            steps_base,  cos_base,
                            out_dir, task_id, prev_task):
    n_layers = len(cos_ortho)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(max(4 * n_layers, 8), 4), sharey=True
    )
    if n_layers == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        if d < len(cos_ortho):
            ax.plot(steps_ortho, cos_ortho[d],
                    color='royalblue', linewidth=1.5,
                    label='Tree_LoRA + Ortho')
        if cos_base and d < len(cos_base):
            ax.plot(steps_base, cos_base[d],
                    color='tomato', linewidth=1.5, linestyle='--',
                    label='Tree_LoRA (baseline)')
        ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
        ax.set_title(f'Layer {d}', fontsize=10)
        ax.set_xlabel('Step')
        if d == 0:
            ax.set_ylabel('Cosine Similarity  g_t · g_k')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Cosine Similarity — Task {task_id} vs prev Task {prev_task}',
        fontsize=13, y=1.02
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'cos_sim_task{task_id}_prev{prev_task}.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_orth_norm(steps_ortho, orth_ortho,
                   steps_base,  orth_base,
                   out_dir, task_id, prev_task):
    n_layers = len(orth_ortho)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(max(4 * n_layers, 8), 4), sharey=False
    )
    if n_layers == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        if d < len(orth_ortho):
            ax.plot(steps_ortho, orth_ortho[d],
                    color='royalblue', linewidth=1.5,
                    label='Tree_LoRA + Ortho')
        if orth_base and d < len(orth_base):
            ax.plot(steps_base, orth_base[d],
                    color='tomato', linewidth=1.5, linestyle='--',
                    label='Tree_LoRA (baseline)')
        ax.set_title(f'Layer {d}', fontsize=10)
        ax.set_xlabel('Step')
        if d == 0:
            ax.set_ylabel('Projection Norm  ||P_k g_t||')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Orthogonal Projection Norm — Task {task_id} vs prev Task {prev_task}',
        fontsize=13, y=1.02
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'orth_norm_task{task_id}_prev{prev_task}.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_orth_loss(steps, orth_loss, out_dir, task_id):
    valid = [(s, v) for s, v in zip(steps, orth_loss) if v is not None]
    if not valid:
        return
    xs, ys = zip(*valid)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, ys, color='darkorange', linewidth=1.8, label='Ortho Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Orthogonal Projection Loss')
    ax.set_title(f'Orthogonal Loss Curve — Task {task_id}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'orth_loss_task{task_id}.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_gradient_heatmap(log, out_dir, task_id, label='ortho'):
    """
    Heatmap: x=step, y=prev_task, colour=mean cosine similarity across layers.
    Useful for seeing which past tasks the current gradient conflicts with.
    """
    if not log:
        return

    n_prev   = max(len(e.get('cos_sim_per_layer', [])) for e in log)
    steps    = [e['step'] for e in log]
    heat     = np.full((n_prev, len(steps)), np.nan)

    for t_idx, entry in enumerate(log):
        cos_data = entry.get('cos_sim_per_layer', [])
        for k, cos_vec in enumerate(cos_data):
            if cos_vec:
                heat[k, t_idx] = float(np.mean(cos_vec))

    fig, ax = plt.subplots(figsize=(min(len(steps) * 0.12 + 2, 16), max(n_prev * 0.6 + 1, 3)))
    im = ax.imshow(heat, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   extent=[steps[0], steps[-1], n_prev - 0.5, -0.5])
    ax.set_xlabel('Step')
    ax.set_ylabel('Previous Task k')
    ax.set_title(f'Mean Cosine Similarity Heatmap — Task {task_id} [{label}]')
    ax.set_yticks(range(n_prev))
    fig.colorbar(im, ax=ax, label='cos(g_t, g_k)')
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'cos_heatmap_task{task_id}_{label}.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved → {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_ortho', required=True,
                        help='vis_log JSON from Tree_LoRA_Ortho')
    parser.add_argument('--log_base',  default=None,
                        help='vis_log JSON from vanilla Tree_LoRA (optional)')
    parser.add_argument('--out_dir',   default='figures')
    parser.add_argument('--task_id',   type=int, default=1,
                        help='Which training task this log belongs to')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    log_ortho = load_log(args.log_ortho)
    log_base  = load_log(args.log_base) if args.log_base else None

    # determine how many previous tasks appear in the log
    n_prev_max = max(
        (len(e.get('cos_sim_per_layer', [])) for e in log_ortho),
        default=0
    )

    for prev_task in range(n_prev_max):
        steps_o, cos_o, orth_o, ol_o = extract_series(log_ortho, prev_task)

        steps_b, cos_b, orth_b = [], [], []
        if log_base:
            steps_b, cos_b, orth_b, _ = extract_series(log_base, prev_task)

        if not steps_o:
            continue

        plot_cosine_similarity(
            steps_o, cos_o, steps_b, cos_b,
            args.out_dir, args.task_id, prev_task
        )
        plot_orth_norm(
            steps_o, orth_o, steps_b, orth_b,
            args.out_dir, args.task_id, prev_task
        )

    plot_orth_loss(
        [e['step'] for e in log_ortho],
        [e.get('orth_loss') for e in log_ortho],
        args.out_dir, args.task_id
    )

    plot_gradient_heatmap(log_ortho, args.out_dir, args.task_id, label='ortho')
    if log_base:
        plot_gradient_heatmap(log_base, args.out_dir, args.task_id, label='base')


if __name__ == '__main__':
    main()