"""Draw the analytical framework and documented sample-selection flow.

These descriptive diagrams contain no newly estimated statistical results.
The layout is reconstructed from the archived figures.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


def box(ax, x, y, width, height, label):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle='round,pad=0.006,rounding_size=0.008',
                          linewidth=0.9, edgecolor='#666666', facecolor='white')
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label,
            ha='center', va='center', fontsize=9)


def arrow(ax, start, end, dashed=False):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '->', 'lw': 0.9, 'color': '#666666',
                            'linestyle': '--' if dashed else '-'})


def export(fig, destination, name):
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ('eps', 'svg'):
        fig.savefig(destination / f'{name}.{extension}', facecolor='white')
    fig.savefig(destination / f'{name}.png', dpi=200, facecolor='white')
    temporary = destination / f'{name}_render.png'
    fig.savefig(temporary, dpi=600, facecolor='white')
    with Image.open(temporary) as im:
        im.convert('RGB').save(destination / f'{name}.tiff',
                               compression='raw', dpi=(600, 600))
    temporary.unlink()
    plt.close(fig)


def main(root):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.hashsalt': 'review-helpfulness-inconsistency'})
    destination = root / 'figures'
    fig, ax = plt.subplots(figsize=(150 / 25.4, 90 / 25.4))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    box(ax, .05, .41, .28, .20, 'Rating–text\ninconsistency')
    box(ax, .67, .41, .28, .20, 'Review helpfulness\nHelpful vote count')
    box(ax, .36, .77, .28, .18, 'Verified Purchase\nSource cue')
    box(ax, .05, .06, .30, .20, 'Directional decomposition\nText more positive\nText more negative')
    box(ax, .64, .06, .32, .20, 'Sensitivity analyses\nProduct type and effects\nMeasurement confidence')
    arrow(ax, (.33, .51), (.67, .51))
    ax.text(.41, .55, 'Association', ha='center', fontsize=8)
    arrow(ax, (.50, .77), (.50, .51), dashed=True)
    ax.text(.52, .69, 'Moderation', fontsize=8)
    arrow(ax, (.19, .26), (.19, .41))
    arrow(ax, (.81, .26), (.81, .41), dashed=True)
    fig.subplots_adjust(left=.03, right=.97, bottom=.03, top=.97)
    export(fig, destination, 'Figure_1_framework')

    fig, ax = plt.subplots(figsize=(150 / 25.4, 100 / 25.4))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    labels = [
        'Category–year balanced base\n480,000 reviews',
        'Placeholder texts excluded\n340 records',
        'Additional exact duplicates excluded\n167 records in 159 groups',
        'Final analysis sample\n479,493 reviews',
    ]
    ys = [.78, .55, .32, .09]
    for y, label in zip(ys, labels):
        box(ax, .15, y, .70, .15, label)
    for upper, lower in zip(ys, ys[1:]):
        arrow(ax, (.50, upper), (.50, lower + .15))
    fig.subplots_adjust(left=.03, right=.97, bottom=.03, top=.97)
    export(fig, destination, 'Figure_2_sample_flow')
    print('Drew Figures 1–2 in EPS, SVG, PNG and 600 dpi RGB TIFF formats.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
