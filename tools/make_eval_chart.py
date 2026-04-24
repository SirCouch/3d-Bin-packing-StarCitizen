"""Generate eval_comparison.png with per-ship ΔSR and ΔUtil bars."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = [
    # (ship, category, prev_sr, curr_sr, prev_util, curr_util)
    ("Freelancer DUR/MIS", "Small", 79.2, 96.7, 44.6, 70.4),
    ("Avenger Renegade", "Small", 100.0, 78.3, 75.0, 56.3),
    ("C1 Spirit", "Small", 95.4, 80.0, 71.6, 66.6),
    ("Hull-A", "Small", 100.0, 100.0, 79.7, 70.3),
    ("Hammerhead", "Small", 89.1, 81.7, 65.3, 66.3),
    ("400i", "Small", 67.4, 82.2, 54.8, 62.6),
    ("Cutlass Black", "Small", 85.8, 91.9, 65.0, 67.5),
    ("Zeus Mk II ES", "Small", 78.3, 97.8, 57.8, 68.1),
    ("Freelancer", "Small", 84.4, 77.7, 56.3, 64.4),
    ("Avenger Titan", "Small", 100.0, 76.7, 75.0, 56.3),
    ("Valkyrie", "Small", 84.2, 77.0, 66.0, 61.3),
    ("Apollo", "Small", 92.7, 100.0, 71.9, 68.8),
    ("Clipper", "Small", 90.0, 96.0, 65.0, 65.0),
    ("Prowler Utility", "Small", 97.5, 100.0, 76.9, 68.8),
    ("Shiv", "Small", 100.0, 85.7, 78.1, 62.5),
    ("Mercury Star Runner", "Medium", 71.5, 85.6, 52.0, 63.8),
    ("Zeus Mk II CL", "Medium", 73.6, 79.5, 42.5, 53.0),
    ("Aquila", "Medium", 72.2, 75.8, 58.8, 62.6),
    ("Corsair", "Medium", 78.0, 80.8, 55.8, 64.4),
    ("Freelancer MAX", "Medium", 72.0, 82.7, 42.2, 63.2),
    ("Constellation Taurus", "Medium", 77.1, 86.7, 48.4, 61.7),
    ("Retaliator Cargo Module", "Medium", 82.5, 84.5, 57.4, 59.7),
    ("RAFT", "Medium", 93.1, 79.8, 74.2, 63.3),
    ("Starfarer/Gemini", "Medium", 62.6, 88.9, 37.0, 62.8),
    ("Andromeda Phoenix", "Medium", 97.5, 81.2, 78.0, 68.3),
    ("Andromeda", "Medium", 77.3, 78.2, 56.5, 64.6),
    ("A2 Hercules", "Medium", 69.8, 73.6, 56.8, 61.3),
    ("Perseus", "Medium", 100.0, 77.3, 79.2, 61.3),
    ("Starlancer MAX", "Medium", 84.8, 87.2, 65.1, 61.8),
    ("Asgard", "Medium", 74.9, 81.5, 55.8, 62.4),
    ("Starlancer TAC", "Medium", 83.6, 96.6, 58.3, 68.3),
    ("C2 Hercules", "Large", 68.6, 88.5, 46.9, 68.8),
    ("890 Jump", "Large", 79.6, 81.4, 64.3, 60.6),
    ("Caterpillar", "Large", 65.6, 75.5, 40.5, 44.3),
    ("M2 Hercules", "Large", 73.0, 74.8, 56.1, 56.7),
    ("Polaris", "Large", 78.7, 90.3, 60.1, 69.4),
    ("Carrack", "Large", 95.3, 100.0, 75.3, 71.7),
    ("Hermes", "Large", 93.0, 83.8, 74.9, 68.2),
    ("Hull-B", "Large", 99.5, 100.0, 79.3, 71.9),
    ("Hull-c", "Large", 81.5, 100.0, 63.4, 72.0),
]

CAT_COLOR = {"Small": "#06d6a0", "Medium": "#118ab2", "Large": "#ef476f"}


def main():
    ships = [d[0] for d in DATA]
    cats = [d[1] for d in DATA]
    d_sr = [d[3] - d[2] for d in DATA]
    d_util = [d[5] - d[4] for d in DATA]

    # Sort by ΔSR descending so wins are at top
    order = sorted(range(len(ships)), key=lambda i: d_sr[i], reverse=True)
    ships = [ships[i] for i in order]
    cats = [cats[i] for i in order]
    d_sr = [d_sr[i] for i in order]
    d_util = [d_util[i] for i in order]
    colors = [CAT_COLOR[c] for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(14, 12), sharey=True)
    y = list(range(len(ships)))

    # Δ Success Rate
    axes[0].barh(y, d_sr, color=colors, edgecolor='#333', linewidth=0.4)
    axes[0].axvline(0, color='#666', linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ships, fontsize=9)
    axes[0].set_xlabel('Δ Success Rate (pp)')
    axes[0].set_title('Change in Success Rate')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # Δ Volume Utilization
    axes[1].barh(y, d_util, color=colors, edgecolor='#333', linewidth=0.4)
    axes[1].axvline(0, color='#666', linewidth=0.8)
    axes[1].set_xlabel('Δ Volume Utilization (pp)')
    axes[1].set_title('Change in Volume Utilization')
    axes[1].grid(axis='x', alpha=0.3)

    # Legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLOR.values()]
    axes[0].legend(legend_handles, CAT_COLOR.keys(), loc='lower right', title='Category')

    fig.suptitle('Eval Comparison — Previous vs Current Run (per-ship Δ, sorted by ΔSR)',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    out = 'eval_comparison.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
