"""Generate high-resolution academic diagrams for the AutoWrangler research paper/proposal."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set global styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def generate_figure1_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # Color palette (Academic clean)
    c_input = "#EBF3FB"
    c_meta = "#E8F5E9"
    c_plan = "#FFF8E1"
    c_exec = "#F3E5F5"
    c_qa = "#FBE9E7"
    c_out = "#E0F2F1"
    border_col = "#37474F"

    # Box styling helper
    def draw_node(x, y, w, h, text, subtext, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5",
                                     facecolor=color, edgecolor=border_col, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, text, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1A237E')
        ax.text(x + w/2, y + h*0.28, subtext, ha='center', va='center', fontsize=7.5, color='#455A64', style='italic')

    # Draw nodes
    draw_node(2, 28, 12, 14, "Input CSV", "Dirty Dataset\n(File on disk)", c_input)
    draw_node(18, 28, 14, 14, "Profiler Node", "Computes Stats &\nParse Rates", c_meta)
    draw_node(36, 28, 14, 14, "Planner Node", "LLM Reasoning\n(Schema Only)", c_plan)
    draw_node(54, 28, 14, 14, "Reviewer Node", "Pydantic Schema\nValidation", c_plan)
    draw_node(72, 28, 14, 14, "Executor Node", "Deterministic\nPandas Runtime", c_exec)
    draw_node(72, 6, 14, 14, "QA Validator", "Null & Anomaly\nVerification", c_qa)
    draw_node(36, 6, 14, 14, "Analyst Node", "Summary Report\n& Observations", c_meta)
    draw_node(2, 6, 14, 14, "Outputs", "Clean Data, Code,\nSHA-256 Passport", c_out)

    # Arrows
    arrow_style = dict(facecolor='#263238', edgecolor='#263238', width=1.0, headwidth=6, shrink=0.08)
    
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#37474F', edgecolor='#37474F', width=1.2, headwidth=5, headlength=6))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 2, label, ha='center', va='center', fontsize=7, fontweight='bold', color='#B71C1C')

    draw_arrow(14, 35, 18, 35)
    draw_arrow(32, 35, 36, 35)
    draw_arrow(50, 35, 54, 35)
    draw_arrow(68, 35, 72, 35)
    draw_arrow(79, 28, 79, 20)  # Executor to QA
    
    # Retry arrow (QA -> Planner)
    ax.annotate('', xy=(43, 28), xytext=(72, 13),
                arrowprops=dict(arrowstyle="->", color='#C62828', lw=1.5, connectionstyle="arc3,rad=-0.25"))
    ax.text(56, 18, "Retry (Restart from Orig CSV)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#C62828')

    # QA -> Analyst (Pass)
    draw_arrow(72, 13, 50, 13)
    ax.text(61, 15, "Approved", ha='center', va='center', fontsize=7.5, color='#2E7D32', fontweight='bold')

    # Analyst -> Outputs
    draw_arrow(36, 13, 16, 13)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure1_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figure1_architecture.png")

def generate_figure2_comparison():
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    categories = ['Execution\nReliability (%)', 'Privacy Guarantee\n(Zero Rows Sent %)', 'Data Schema\nPreservation (%)', 'Token Efficiency\n(Norm % Saved)']
    autowrangler = [100, 100, 100, 92]
    unconstrained = [68, 0, 72, 8]

    x = range(len(categories))
    width = 0.35

    rects1 = ax.bar([i - width/2 for i in x], autowrangler, width, label='AutoWrangler (Constrained DAG)', color='#1976D2', edgecolor='#0D47A1')
    rects2 = ax.bar([i + width/2 for i in x], unconstrained, width, label='Unconstrained LLM Code Generation', color='#E57373', edgecolor='#B71C1C')

    ax.set_ylabel('Score / Percentage (%)', fontsize=9.5, fontweight='bold')
    ax.set_title('Figure 2: Empirical Performance vs. Unconstrained Code Generation Baselines', fontsize=10.5, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, fontsize=8.5, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 2, f'{int(h)}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#0D47A1')

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 2, f'{int(h)}%', ha='center', va='bottom', fontsize=8, color='#B71C1C')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figure2_comparison.png")

if __name__ == "__main__":
    generate_figure1_architecture()
    generate_figure2_comparison()

