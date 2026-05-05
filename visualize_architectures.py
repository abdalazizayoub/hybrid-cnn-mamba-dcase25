#!/usr/bin/env python3
"""
Saves 4 PNG architecture diagrams using PlotNeuralNet-style 3D block visuals:
  hybrid_mamba.png   – blue
  hybrid_xlstm.png   – green
  hybrid_gru.png     – orange
  all_architectures.png – all three side-by-side

Also writes PlotNeuralNet LaTeX (.tex) files to ./diagrams/ for PDF compilation.

Usage:
    conda run -n aum python visualize_architectures.py
    python visualize_architectures.py --embed-dim 64 --depth 4
"""

import argparse
import colorsys
import io
import contextlib
import os
import shutil
import subprocess
import sys

ROOT     = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(ROOT, "diagrams")
PLOTNN   = os.path.join(ROOT, "PlotNeuralNet")
os.makedirs(DIAG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────

def _pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pkgs], check=True)

def ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[deps] Installing matplotlib …")
        _pip_install("matplotlib")

def ensure_plotneuralnet():
    if not os.path.isdir(PLOTNN):
        print("[PlotNeuralNet] Cloning repository …")
        subprocess.run(
            ["git", "clone", "https://github.com/HarisIqbal88/PlotNeuralNet.git", PLOTNN],
            check=True,
        )
    if PLOTNN not in sys.path:
        sys.path.insert(0, PLOTNN)

# ─────────────────────────────────────────────────────────────────
# Color palettes
# ─────────────────────────────────────────────────────────────────

PALETTES = {
    "mamba": {
        "conv":     "#1976D2",
        "bridge":   "#0288D1",
        "temporal": "#0D47A1",
        "norm":     "#42A5F5",
        "pool":     "#64B5F6",
        "cls":      "#1565C0",
        "input":    "#BBDEFB",
        "bg":       "#E3F2FD",
    },
    "xlstm": {
        "conv":     "#388E3C",
        "bridge":   "#00897B",
        "temporal": "#1B5E20",
        "norm":     "#66BB6A",
        "pool":     "#81C784",
        "cls":      "#2E7D32",
        "input":    "#C8E6C9",
        "bg":       "#E8F5E9",
    },
    "gru": {
        "conv":     "#F57C00",
        "bridge":   "#FF8F00",
        "temporal": "#BF360C",
        "norm":     "#FFA726",
        "pool":     "#FFB74D",
        "cls":      "#E65100",
        "input":    "#FFE0B2",
        "bg":       "#FFF3E0",
    },
}

# LaTeX color palettes for PlotNeuralNet .tex files
def _cor_mamba():
    return r"""
\def\ConvColor{rgb,255:red,66;green,133;blue,244}
\def\ConvReluColor{rgb,255:red,30;green,136;blue,229}
\def\PoolColor{rgb,255:red,144;green,202;blue,249}
\def\UnpoolColor{rgb,255:red,100;green,181;blue,246}
\def\FcColor{rgb,255:red,21;green,101;blue,192}
\def\FcReluColor{rgb,255:red,13;green,71;blue,161}
\def\SoftmaxColor{rgb,255:red,187;green,222;blue,251}
"""

def _cor_xlstm():
    return r"""
\def\ConvColor{rgb,255:red,67;green,160;blue,71}
\def\ConvReluColor{rgb,255:red,46;green,125;blue,50}
\def\PoolColor{rgb,255:red,165;green,214;blue,167}
\def\UnpoolColor{rgb,255:red,129;green,199;blue,132}
\def\FcColor{rgb,255:red,27;green,94;blue,32}
\def\FcReluColor{rgb,255:red,0;green,77;blue,26}
\def\SoftmaxColor{rgb,255:red,200;green,230;blue,201}
"""

def _cor_gru():
    return r"""
\def\ConvColor{rgb,255:red,251;green,140;blue,0}
\def\ConvReluColor{rgb,255:red,245;green,124;blue,0}
\def\PoolColor{rgb,255:red,255;green,204;blue,128}
\def\UnpoolColor{rgb,255:red,255;green,183;blue,77}
\def\FcColor{rgb,255:red,230;green,81;blue,0}
\def\FcReluColor{rgb,255:red,191;green,54;blue,12}
\def\SoftmaxColor{rgb,255:red,255;green,236;blue,179}
"""

# ─────────────────────────────────────────────────────────────────
# PlotNeuralNet .tex generation
# ─────────────────────────────────────────────────────────────────

def _plotnn_cnn_layers():
    from pycore.tikzeng import to_Conv, to_connection
    return [
        to_Conv("input", 256, 1,  offset="(0,0,0)",     to="(0,0,0)",
                height=40, depth=8, width=2,
                caption="Input\\\\1{\\texttimes}256{\\texttimes}33"),
        to_Conv("conv1", 128, 8,  offset="(2.5,0,0)",   to="(input-east)",
                height=34, depth=8, width=2,  caption="ConvBlock 1\\\\8ch+SE"),
        to_connection("input",  "conv1"),
        to_Conv("conv2", 64,  16, offset="(2,0,0)",     to="(conv1-east)",
                height=26, depth=8, width=3,  caption="ConvBlock 2\\\\16ch+SE"),
        to_connection("conv1",  "conv2"),
        to_Conv("conv3", 32,  16, offset="(2,0,0)",     to="(conv2-east)",
                height=18, depth=8, width=3,  caption="ConvBlock 3\\\\16ch+SE"),
        to_connection("conv2",  "conv3"),
        to_Conv("conv4", 16,  32, offset="(2,0,0)",     to="(conv3-east)",
                height=12, depth=8, width=4,  caption="ConvBlock 4\\\\32ch+SE"),
        to_connection("conv3",  "conv4"),
    ]

def _plotnn_tail_mamba(embed_dim, depth):
    from pycore.tikzeng import to_Conv, to_Pool, to_SoftMax, to_connection
    return [
        to_Conv("bridge", 16, embed_dim, offset="(2,0,0)",   to="(conv4-east)",
                height=12, depth=2, width=5,  caption="Bridge\\\\Linear+LN+GELU"),
        to_connection("conv4",   "bridge"),
        to_Conv("mamba",  16, embed_dim, offset="(2.5,0,0)", to="(bridge-east)",
                height=12, depth=2, width=6,
                caption=f"Mamba SSM\\\\{depth} block{'s' if depth>1 else ''}"),
        to_connection("bridge",  "mamba"),
        to_Conv("fnorm",  16, embed_dim, offset="(2,0,0)",   to="(mamba-east)",
                height=12, depth=2, width=2,  caption="LayerNorm"),
        to_connection("mamba",   "fnorm"),
        to_Pool("gap",  offset="(2,0,0)",   to="(fnorm-east)",  height=5, depth=2, width=1),
        to_connection("fnorm",   "gap"),
        to_SoftMax("cls", s_filer=10, offset="(2,0,0)", to="(gap-east)", caption="Classifier"),
        to_connection("gap",     "cls"),
    ]

def _plotnn_tail_xlstm(embed_dim, depth):
    from pycore.tikzeng import to_Conv, to_Pool, to_SoftMax, to_connection
    return [
        to_Conv("bridge", 16, embed_dim, offset="(2,0,0)",   to="(conv4-east)",
                height=12, depth=2, width=5,
                caption="Bridge\\\\Linear+LN+GELU+Drop"),
        to_connection("conv4",   "bridge"),
        to_Conv("xlstm",  16, embed_dim, offset="(2.5,0,0)", to="(bridge-east)",
                height=12, depth=2, width=7,
                caption=f"xLSTM Stack\\\\mLSTM+sLSTM {depth}blk"),
        to_connection("bridge",  "xlstm"),
        to_Conv("fnorm",  16, embed_dim, offset="(2,0,0)",   to="(xlstm-east)",
                height=12, depth=2, width=2,  caption="LayerNorm"),
        to_connection("xlstm",   "fnorm"),
        to_Pool("gap",  offset="(2,0,0)",   to="(fnorm-east)",  height=5, depth=2, width=1),
        to_connection("fnorm",   "gap"),
        to_Conv("drop",   1,  embed_dim, offset="(1.5,0,0)", to="(gap-east)",
                height=5,  depth=2, width=1,  caption="Drop 0.3"),
        to_connection("gap",     "drop"),
        to_SoftMax("cls", s_filer=10, offset="(2,0,0)", to="(drop-east)", caption="Classifier"),
        to_connection("drop",    "cls"),
    ]

def _plotnn_tail_gru(embed_dim, depth):
    from pycore.tikzeng import to_Conv, to_Pool, to_SoftMax, to_connection
    return [
        to_Conv("bridge", 16, embed_dim, offset="(2,0,0)",   to="(conv4-east)",
                height=12, depth=2, width=5,  caption="Bridge\\\\Linear+LN+GELU"),
        to_connection("conv4",   "bridge"),
        to_Conv("gru",    16, embed_dim, offset="(2.5,0,0)", to="(bridge-east)",
                height=12, depth=2, width=6,
                caption=f"GRU\\\\{depth} layer{'s' if depth>1 else ''}, uni-dir"),
        to_connection("bridge",  "gru"),
        to_Conv("fnorm",  16, embed_dim, offset="(2,0,0)",   to="(gru-east)",
                height=12, depth=2, width=2,  caption="LayerNorm"),
        to_connection("gru",     "fnorm"),
        to_Pool("gap",  offset="(2,0,0)",   to="(fnorm-east)",  height=5, depth=2, width=1),
        to_connection("fnorm",   "gap"),
        to_SoftMax("cls", s_filer=10, offset="(2,0,0)", to="(gap-east)", caption="Classifier"),
        to_connection("gap",     "cls"),
    ]

_TEX_SPECS = {
    "hybrid_mamba": (_cor_mamba, _plotnn_cnn_layers, _plotnn_tail_mamba),
    "hybrid_xlstm": (_cor_xlstm, _plotnn_cnn_layers, _plotnn_tail_xlstm),
    "hybrid_gru":   (_cor_gru,   _plotnn_cnn_layers, _plotnn_tail_gru),
}

def generate_tex_files(embed_dim, depth):
    ensure_plotneuralnet()
    from pycore.tikzeng import to_head, to_begin, to_end, to_generate

    has_latex = shutil.which("pdflatex") is not None
    print(f"[PlotNeuralNet] pdflatex {'found' if has_latex else 'NOT found – .tex only'}")

    for name, (cor_fn, cnn_fn, tail_fn) in _TEX_SPECS.items():
        arch = (
            [to_head(PLOTNN), cor_fn(), to_begin()]
            + cnn_fn()
            + tail_fn(embed_dim, depth)
            + [to_end()]
        )
        tex_path = os.path.join(DIAG_DIR, name + ".tex")
        with contextlib.redirect_stdout(io.StringIO()):
            to_generate(arch, tex_path)
        print(f"  → {tex_path}")

        if has_latex:
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", DIAG_DIR, tex_path],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                print(f"  → {os.path.join(DIAG_DIR, name + '.pdf')}")
            else:
                print(f"  pdflatex error: {r.stderr[-400:]}")

    if not has_latex:
        print("  (install texlive to compile to PDF)")

# ─────────────────────────────────────────────────────────────────
# PlotNeuralNet-style 3D block drawing with matplotlib
# ─────────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    return tuple(int(h.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))

def _adjust(color_hex, factor):
    r, g, b = _hex_to_rgb(color_hex)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = min(1.0, v * factor)
    s = max(0.0, s * (0.65 if factor > 1 else 1.0))
    return colorsys.hsv_to_rgb(h, s, v)

def _draw_block(ax, x, y, w, h, d, hex_color):
    """3D box — front + top + right face. No text inside."""
    from matplotlib.patches import Polygon
    px, py = d * 0.5, d * 0.32
    front = _hex_to_rgb(hex_color)
    top   = _adjust(hex_color, 1.40)
    right = _adjust(hex_color, 0.58)
    kw = dict(closed=True, edgecolor="white", linewidth=0.7, zorder=3)
    ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],         facecolor=front, **kw))
    ax.add_patch(Polygon([(x,y+h),(x+w,y+h),(x+w+px,y+h+py),(x+px,y+h+py)], facecolor=top,   **kw))
    ax.add_patch(Polygon([(x+w,y),(x+w+px,y+py),(x+w+px,y+h+py),(x+w,y+h)], facecolor=right, **kw))

def _arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="#666",
                                lw=1.1, mutation_scale=9), zorder=4)

def draw_arch(ax, title, pal, temporal_label, temporal_sub,
              bridge_sub, embed_dim, depth):
    """Render one architecture as PlotNeuralNet-style 3D blocks.
    All text lives outside the blocks: dim-labels above, name+detail below."""

    GAP    = 0.65    # gap between block right-edges
    D_CNN  = 0.72    # 3-D depth for CNN blocks
    D_REST = 0.38    # 3-D depth for bridge / temporal / output blocks
    Y0     = 2.4     # bottom of blocks; room below reserved for text

    blocks = [
        # (name, detail, dim_label, width, height, d3, color_key)
        ("Input",       "1×256×33",              "1 ch",          0.5,  5.0, D_CNN,  "input"),
        ("ConvBlock 1", "8 ch · stride (2,1)",   "8 ch",          0.6,  4.2, D_CNN,  "conv"),
        ("ConvBlock 2", "16 ch · stride (2,1)",  "16 ch",         0.8,  3.2, D_CNN,  "conv"),
        ("ConvBlock 3", "16 ch · stride (2,1)",  "16 ch",         0.8,  2.2, D_CNN,  "conv"),
        ("ConvBlock 4", "32 ch · stride (2,1)",  "32 ch",         1.05, 1.5, D_CNN,  "conv"),
        ("Bridge",      bridge_sub,               f"{embed_dim} d", 0.95, 1.5, D_REST, "bridge"),
        (temporal_label, temporal_sub,            f"{embed_dim} d", 1.35, 1.5, D_REST, "temporal"),
        ("LayerNorm",   "",                       "",              0.45, 1.5, D_REST, "norm"),
        ("Avg Pool",    "freq → vector",          "",              0.55, 0.9, D_REST, "pool"),
        ("Classifier",  "→ 10 classes",           "",              0.75, 0.9, D_REST, "cls"),
    ]
    if "xLSTM" in temporal_label:
        blocks.insert(9, ("Dropout", "p = 0.3", "", 0.42, 0.9, D_REST, "norm"))

    # x positions: each block starts after (previous block front+3D-right-face + GAP)
    xs = []
    x = 0.0
    for (_, _, _, w, _, d3, _) in blocks:
        xs.append(x)
        x += w + d3 * 0.5 + GAP

    total_w = x - GAP + 0.5
    max_h   = max(b[4] for b in blocks)

    # Canvas: Y0 downward ≈ 2.4 units for text; upward = max_h + perspective + title
    y_bot = -(Y0)
    y_top = max_h + D_CNN * 0.32 + 1.6      # perspective top + title room
    ax.set_xlim(-0.4, total_w)
    ax.set_ylim(y_bot, y_top)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(pal["bg"])

    # ── Title ──────────────────────────────────────────────────
    ax.text(total_w / 2, y_top - 0.15, title,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=_adjust(pal["cls"], 0.75), zorder=6)

    # ── Blocks + labels ────────────────────────────────────────
    for i, (name, detail, dim, w, h, d3, ckey) in enumerate(blocks):
        xi = xs[i]
        _draw_block(ax, xi, 0, w, h, d3, pal[ckey])

        cx = xi + w / 2          # horizontal center of the front face

        # Dimension annotation just above the top face
        if dim:
            ax.text(cx, h + d3 * 0.32 + 0.12, dim,
                    ha="center", va="bottom", fontsize=5.5, fontweight="bold",
                    color=_adjust(pal[ckey], 0.62), zorder=5)

        # Arrow from previous block's right edge to this block's left edge
        if i > 0:
            _, _, _, pw, ph, pd3, _ = blocks[i - 1]
            x_tail = xs[i-1] + pw + pd3 * 0.5
            y_arr  = min(ph, h) / 2
            _arrow(ax, x_tail, xi, y_arr)

        # ── Text BELOW the block (on the background — always readable) ──
        # name (bold)
        ax.text(cx, -0.18, name,
                ha="center", va="top", fontsize=7.2, fontweight="bold",
                color="#1a1a1a", zorder=5)
        # detail (regular, smaller)
        if detail:
            ax.text(cx, -0.72, detail,
                    ha="center", va="top", fontsize=5.8, color="#444444",
                    zorder=5)


def generate_pngs(embed_dim, depth):
    ensure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cwd = os.getcwd()

    specs = [
        ("hybrid_mamba", "Hybrid CNN-Mamba",
         "mamba",
         "Mamba SSM",    f"d_model={embed_dim}, d_state=16",
         "Linear → LN → GELU"),
        ("hybrid_xlstm", "Hybrid CNN-xLSTM",
         "xlstm",
         "xLSTM Stack",  f"mLSTM+sLSTM ×{depth}",
         "Linear → LN → GELU → Drop"),
        ("hybrid_gru",   "Hybrid CNN-GRU",
         "gru",
         "GRU",          f"{depth} layer{'s' if depth>1 else ''}, uni-dir",
         "Linear → LN → GELU"),
    ]

    figs = []
    for filename, title, key, tlabel, tsub, bsub in specs:
        # set_aspect("equal") means matplotlib sizes the axes to the data range;
        # use a wide figure so the horizontal layout isn't squashed
        fig, ax = plt.subplots(figsize=(18, 5.5))
        fig.patch.set_facecolor(PALETTES[key]["bg"])
        ax.set_facecolor(PALETTES[key]["bg"])
        draw_arch(ax, title, PALETTES[key], tlabel, tsub, bsub, embed_dim, depth)
        out = os.path.join(cwd, filename + ".png")
        fig.savefig(out, dpi=180, bbox_inches="tight",
                    facecolor=PALETTES[key]["bg"])
        plt.close(fig)
        print(f"  → {out}")
        figs.append((filename, title, key, tlabel, tsub, bsub))

    # Combined overview  – stack the three rows vertically
    fig, axes = plt.subplots(3, 1, figsize=(18, 16))
    fig.patch.set_facecolor("#F5F5F5")
    for ax, (filename, title, key, tlabel, tsub, bsub) in zip(axes, figs):
        ax.set_facecolor(PALETTES[key]["bg"])
        draw_arch(ax, title, PALETTES[key], tlabel, tsub, bsub, embed_dim, depth)
    plt.suptitle("Hybrid CNN Architectures – DCASE 2025",
                 fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout(pad=0.8, h_pad=2.5)
    out = os.path.join(cwd, "all_architectures.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#F5F5F5")
    plt.close(fig)
    print(f"  → {out}  (combined)")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--depth",     type=int, default=2)
    parser.add_argument("--no-latex",  action="store_true",
                        help="Skip PlotNeuralNet .tex generation")
    args = parser.parse_args()

    print("=" * 58)
    print("  Architecture Diagrams – Hybrid CNN-{Mamba|xLSTM|GRU}")
    print("=" * 58)

    print("\n[PNG] Drawing PlotNeuralNet-style diagrams …")
    generate_pngs(args.embed_dim, args.depth)

    if not args.no_latex:
        print("\n[PlotNeuralNet] Writing LaTeX .tex files …")
        try:
            generate_tex_files(args.embed_dim, args.depth)
        except Exception as e:
            print(f"  failed: {e}")

    print(f"\n{'='*58}")
    print(f"  PNGs saved to: {os.getcwd()}/")
    print(f"  LaTeX files:   {DIAG_DIR}/")
    print("=" * 58)


if __name__ == "__main__":
    main()
