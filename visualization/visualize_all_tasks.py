"""
visualize_all_tasks.py
──────────────────────
Visualize gradient statistics across ALL tasks từ Tree_LoRA_Ortho vis_log.

Usage:
    python visualization/visualize_all_tasks.py \
        --log_dir ./outputs_LLM-CL/cl/Llama-3.2-1B-Instruct/Tree_LoRA_Ortho_0408_123345 \
        --out_dir figures/all_tasks
"""

import argparse
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TASK_NAMES = ["C-STANCE", "FOMC", "MeetingBank", "Py150",
              "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

def load_all_logs(log_dir):
    files = sorted(glob.glob(os.path.join(log_dir, 'vis_log_task*.json')))
    logs = {}
    for fpath in files:
        tid = int(fpath.split('task')[-1].replace('.json', ''))
        with open(fpath) as f:
            data = json.load(f)
        logs[tid] = data
    return logs


def extract_final_stats(logs):
    """
    Trả về dict:
      task_id -> {
        'vs_task_k': { k: {'avg_cos', 'avg_orth', 'min_cos', 'max_cos'} },
        'orth_loss_curve': [float],
        'steps': [int]
      }
    """
    result = {}
    for tid, log in logs.items():
        if not log:
            continue
        vs = {}
        for entry in log:
            cos_data  = entry.get('cos_sim_per_layer', [])
            orth_data = entry.get('orth_norm_per_layer', [])
            for k in range(len(cos_data)):
                if k not in vs:
                    vs[k] = {'cos': [], 'orth': []}
                vs[k]['cos'].append(np.mean(cos_data[k]))
                vs[k]['orth'].append(np.mean(orth_data[k]))

        orth_loss_curve = [e.get('orth_loss', None) for e in log]
        steps           = [e.get('step', i*50) for i, e in enumerate(log)]

        result[tid] = {
            'vs': vs,
            'orth_loss_curve': orth_loss_curve,
            'steps': steps
        }
    return result


# ── Plot 1: Cosine similarity heatmap (final value, task_t vs task_k) ────────

def plot_cos_heatmap_final(stats, out_dir):
    max_task = max(stats.keys())
    mat = np.full((max_task + 1, max_task + 1), np.nan)

    for tid, s in stats.items():
        for k, vdata in s['vs'].items():
            if vdata['cos']:
                mat[tid, k] = vdata['cos'][-1]   # final value

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, label='cos(g_t, g_k)')

    tnames = [TASK_NAMES[i] if i < len(TASK_NAMES) else str(i)
              for i in range(max_task + 1)]
    ax.set_xticks(range(max_task + 1))
    ax.set_yticks(range(max_task + 1))
    ax.set_xticklabels(tnames, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(tnames, fontsize=8)
    ax.set_xlabel('Previous Task k  (g_k)')
    ax.set_ylabel('Current Task t  (g_t)')
    ax.set_title('Final Cosine Similarity: g_t vs g_k\n(blue=similar, red=conflicting)',
                 fontsize=12)

    # annotate values
    for i in range(max_task + 1):
        for j in range(max_task + 1):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if abs(mat[i,j]) > 0.5 else 'black')

    fig.tight_layout()
    path = os.path.join(out_dir, 'cos_similarity_heatmap_final.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {path}')


# ── Plot 2: Orthogonal loss curve per task ────────────────────────────────────

def plot_orth_loss_curves(stats, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats)))

    for (tid, s), color in zip(sorted(stats.items()), colors):
        curve = [v for v in s['orth_loss_curve'] if v is not None]
        steps = s['steps'][:len(curve)]
        if not curve:
            continue
        label = TASK_NAMES[tid] if tid < len(TASK_NAMES) else f'Task {tid}'
        # offset steps to show continuous training
        ax.plot(range(len(curve)), curve, color=color,
                linewidth=1.8, label=f'Task {tid}: {label}')

    ax.set_xlabel('Vis step (within task)')
    ax.set_ylabel('Orthogonal Projection Loss')
    ax.set_title('Orthogonal Loss Curve — All Tasks', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'orth_loss_all_tasks.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {path}')


# ── Plot 3: Cos similarity trend over training steps (per task) ───────────────

def plot_cos_trend_per_task(stats, out_dir):
    n_tasks = len(stats)
    if n_tasks == 0:
        return

    task_ids = sorted(stats.keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axes = axes.flatten()

    for ax_idx, tid in enumerate(task_ids):
        ax = axes[ax_idx]
        s  = stats[tid]
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(s['vs']), 1)))

        for (k, vdata), color in zip(sorted(s['vs'].items()), colors):
            label = TASK_NAMES[k] if k < len(TASK_NAMES) else f'Task {k}'
            ax.plot(vdata['cos'], color=color, linewidth=1.5,
                    label=f'vs {label}')

        ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            f'Task {tid}: {TASK_NAMES[tid] if tid < len(TASK_NAMES) else ""}',
            fontsize=9
        )
        ax.set_xlabel('Vis step', fontsize=8)
        if ax_idx % 4 == 0:
            ax.set_ylabel('cos(g_t, g_k)', fontsize=8)
        ax.legend(fontsize=6, ncol=1)
        ax.grid(True, alpha=0.25)

    # hide unused subplots
    for i in range(len(task_ids), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Cosine Similarity Trend During Training — Per Task', fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, 'cos_trend_all_tasks.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {path}')


# ── Plot 4: Orth norm trend ───────────────────────────────────────────────────

def plot_orth_norm_per_task(stats, out_dir):
    task_ids = sorted(stats.keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax_idx, tid in enumerate(task_ids):
        ax = axes[ax_idx]
        s  = stats[tid]
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(s['vs']), 1)))

        for (k, vdata), color in zip(sorted(s['vs'].items()), colors):
            label = TASK_NAMES[k] if k < len(TASK_NAMES) else f'Task {k}'
            ax.plot(vdata['orth'], color=color, linewidth=1.5,
                    label=f'vs {label}')

        ax.set_title(
            f'Task {tid}: {TASK_NAMES[tid] if tid < len(TASK_NAMES) else ""}',
            fontsize=9
        )
        ax.set_xlabel('Vis step', fontsize=8)
        if ax_idx % 4 == 0:
            ax.set_ylabel('||P_k · g_t||', fontsize=8)
        ax.legend(fontsize=6, ncol=1)
        ax.grid(True, alpha=0.25)

    for i in range(len(task_ids), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Orthogonal Projection Norm ||P_k·g_t|| — Per Task', fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, 'orth_norm_all_tasks.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {path}')


# ── Plot 5: Summary bar chart — avg cos_sim per (task_t, task_k) pair ─────────

def plot_summary_bar(stats, out_dir):
    pairs  = []
    values = []

    for tid in sorted(stats.keys()):
        s = stats[tid]
        for k in sorted(s['vs'].keys()):
            cos_list = s['vs'][k]['cos']
            if cos_list:
                t_name = TASK_NAMES[tid] if tid < len(TASK_NAMES) else str(tid)
                k_name = TASK_NAMES[k]   if k  < len(TASK_NAMES) else str(k)
                pairs.append(f'T{tid}→T{k}\n({t_name[:6]}→{k_name[:6]})')
                values.append(np.mean(cos_list))

    if not pairs:
        return

    colors = ['#2196F3' if v >= 0 else '#F44336' for v in values]
    fig, ax = plt.subplots(figsize=(max(len(pairs) * 0.55 + 1, 8), 5))
    bars = ax.bar(range(len(pairs)), values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=7, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Mean cos(g_t, g_k) for All Task Pairs\n(blue=similar, red=conflicting)',
                 fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.03 if val >= 0 else -0.06),
                f'{val:.2f}', ha='center', va='bottom', fontsize=6)

    fig.tight_layout()
    path = os.path.join(out_dir, 'cos_summary_bar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--out_dir', default='figures/all_tasks')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading logs from {args.log_dir} ...')
    logs  = load_all_logs(args.log_dir)
    stats = extract_final_stats(logs)
    print(f'Found {len(stats)} tasks with data: {sorted(stats.keys())}')

    plot_cos_heatmap_final(stats, args.out_dir)
    plot_orth_loss_curves(stats, args.out_dir)
    plot_cos_trend_per_task(stats, args.out_dir)
    plot_orth_norm_per_task(stats, args.out_dir)
    plot_summary_bar(stats, args.out_dir)

    print(f'\nDone! All figures saved to {args.out_dir}/')


if __name__ == '__main__':
    main()