import csv
import json
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.text import Text as pltText

from eval.cat_graphs import (
    agg_values_to_pcts,
    category_sort,
    colourful,
    convert_raf_ethnicity,
    greyscale,
)
from raf_utils import AGES, GENDERS, RACES
from utils import RAF_DB_EMOTIONS
from utils.bias_categories import (
    AGE_CATEGORIES,
    ETHNICITY_CATEGORIES,
    GENDER_CATEGORIES,
)

AGE_CATEGORIES = category_sort(AGE_CATEGORIES)
ETHNICITY_CATEGORIES = category_sort(ETHNICITY_CATEGORIES)
GENDER_CATEGORIES = category_sort(GENDER_CATEGORIES)

BIASES = ["age", "ethnicity", "gender"]

rc_fonts = {
    "text.usetex": True,
    "text.latex.preamble": "\n".join(
        [r"\usepackage{libertine}", r"\usepackage[libertine]{newtxmath}"]
    ),
}
mpl.rcParams.update(rc_fonts)


def plot(
    orig_data_fp,
    collected_data_fp,
    title,
    save_fig="",
    category_values: Dict[str, List[str]] = {
        "age": AGE_CATEGORIES,
        "ethnicity": ETHNICITY_CATEGORIES,
        "gender": GENDER_CATEGORIES,
    },
    legend_bboxes=[(1.05, 1), (1.05, 0.8), (1.05, 0.6)],
    *,
    in_colour: bool = True,
    use_hatch: bool = False,
):
    with open(orig_data_fp) as f1, open(collected_data_fp) as f2:
        orig_data = json.load(f1)
        coll_data = json.load(f2)

    orig_values_to_plot = [
        agg_values_to_pcts(orig_data[str(i)]["categories"]) for i in range(1, 8)
    ]

    coll_values_to_plot = [
        agg_values_to_pcts(coll_data[str(i)]["categories"]) for i in range(1, 8)
    ]

    colours = colourful if in_colour else greyscale

    colour_map = {
        bias: {
            cat: colours[(i if bias == "age" or in_colour else i * 2) % len(colours)]
            for i, cat in enumerate(categories)
        }
        for bias, categories in category_values.items()
    }

    hatches = ["/", "\\", "+", "x", ".", "*"]

    hatch_map = {
        bias: {cat: hatches[i % len(hatches)] for i, cat in enumerate(categories)}
        for bias, categories in category_values.items()
    }

    # Plot the data

    TICK_MULTIPLIER = 4

    # top will have 3 emotions, bottom will have 3
    x_top = np.arange(4) * TICK_MULTIPLIER
    x_bot = np.arange(3) * TICK_MULTIPLIER

    fig, (top_ax, bot_ax) = plt.subplots(2, 1, figsize=(16, 10))

    top_ax.set_ylim(0, 105)
    bot_ax.set_ylim(0, 105)

    top_texts: List[List[pltText]] = []
    bot_texts: List[List[pltText]] = []

    OFFSET = 1.1

    for em_ind, (orig_emotion, coll_emotion) in enumerate(
        zip(orig_values_to_plot, coll_values_to_plot), start=0
    ):
        for bias_type, offset in zip(BIASES, [-OFFSET, 0, OFFSET]):
            # print(f"{em_ind} - {bias_type}")
            orig_ys = np.array(
                [
                    orig_emotion[bias_type].get(cat.lower(), 0)
                    for cat in category_values[bias_type]
                ]
            )
            coll_ys = np.array(
                [
                    coll_emotion[bias_type].get(cat.lower(), 0)
                    for cat in category_values[bias_type]
                ]
            )
            # if em_ind < 3:
            #     ax = top_ax
            #     texts = top_texts
            #     bar_loc = em_ind
            #     width = 0.35
            #     side_off = 0.22
            # else:
            #     ratio = 0.48 / 0.35
            #     ax = bot_ax
            #     texts = bot_texts
            #     bar_loc = em_ind - 3
            #     width = 0.4
            #     side_off = 0.2 * ratio
            #     offset *= 1.12

            if em_ind < 4:
                ratio = 0.48 / 0.35
                ax = top_ax
                texts = top_texts
                bar_loc = em_ind
                width = 0.4
                side_off = 0.2 * ratio
                offset *= 1.12
            else:
                ax = bot_ax
                texts = bot_texts
                bar_loc = em_ind - 4
                width = 0.35
                side_off = 0.22

            def label_bar(bars, ys, orig=True):
                prev_x = -1
                labs: List[pltText] = []
                for bar, val in zip(bars, ys):
                    if val == 0:
                        continue
                    label_x = bar.get_x() + bar.get_width() / 2
                    label_y = bar.get_height() + bar.get_y()
                    t = ax.text(
                        label_x,
                        label_y,
                        f"{val:.2f}" if val != 100 else "100",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        bbox=dict(
                            facecolor=(1, 1, 1, 0.5),
                            edgecolor="none",
                            # pad=0.01,
                            boxstyle="round,pad=0.01,rounding_size=0.2",
                        ),
                    )
                    if prev_x == label_x or prev_x == -1:
                        labs.append(t)
                    else:
                        texts.append(labs)
                        labs = []

                    prev_x = label_x

                if len(labs) > 0:
                    texts.append(labs)

                ax.text(
                    bars[0].get_x() + bars[0].get_width() / 2,
                    -3,
                    "O" if orig else "C",
                    ha="center",
                    va="center",
                    fontsize=14,
                    rotation=0,
                    fontweight="bold",
                )

            base_x = bar_loc * TICK_MULTIPLIER + offset

            # print(ys)
            orig_bars = ax.bar(
                base_x - side_off,
                orig_ys,
                width=width,
                label=[
                    f"{em_ind}:{bias_type}:{cat}" for cat in category_values[bias_type]
                ],
                bottom=np.cumsum(np.insert(orig_ys, 0, 0)[:-1]),
                color=[v for _, v in colour_map[bias_type].items()],
                hatch=[v for _, v in hatch_map[bias_type].items()]
                if use_hatch
                else None,
            )

            label_bar(orig_bars, orig_ys)

            coll_bars = ax.bar(
                base_x + side_off,
                coll_ys,
                width=width,
                # label=[
                #     f"{em_ind}:{bias_type}:{cat}" for cat in category_values[bias_type]
                # ],
                bottom=np.cumsum(np.insert(coll_ys, 0, 0)[:-1]),
                color=[v for _, v in colour_map[bias_type].items()],
                hatch=[v for _, v in hatch_map[bias_type].items()]
                if use_hatch
                else None,
            )

            label_bar(coll_bars, coll_ys, orig=False)

            ax.text(
                base_x,
                -7,
                bias_type.title(),
                ha="center",
                va="center",
                fontsize=14,
                rotation=0,
            )

    def config_ax(ax: Axes, x: np.ndarray, sl: slice):
        if len(sl.indices(6)) == 4:
            ax.set_xlabel("Emotions", fontsize=18)
        ax.set_xticks(x)
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True,
            labelsize=18,
            pad=25,
        )
        ax.tick_params(axis="y", labelsize=18)
        ax.set_xticklabels(RAF_DB_EMOTIONS[sl])
        # for tl in ax.xaxis.get_majorticklabels():
        #     tl.set_y(tl.get_position()[1] - 0.05)

        ax.set_ylabel(r"Proportions / \%", fontsize=20)

        if title != "":
            ax.set_title(title)

    config_ax(top_ax, x_top, slice(0, 4))
    config_ax(bot_ax, x_bot, slice(4, None))

    # fig.legend()

    handles, labels = top_ax.get_legend_handles_labels()

    age_handles = [
        h for h, lab in zip(handles, labels) if "age" in lab if lab.startswith("0:")
    ]
    age_labels = [
        lab.split(":")[2].strip().title()
        for lab in labels
        if "age" in lab
        if lab.startswith("0:")
    ]
    ethnicity_handles = [
        h
        for h, lab in zip(handles, labels)
        if "ethnicity" in lab
        if lab.startswith("0:")
    ]
    ethnicity_labels = [
        convert_raf_ethnicity(lab.split(":")[2].strip())
        for lab in labels
        if "ethnicity" in lab
        if lab.startswith("0:")
    ]
    gender_handles = [
        h for h, lab in zip(handles, labels) if "gender" in lab if lab.startswith("0:")
    ]
    gender_labels = [
        lab.split(":")[2].strip().title()
        for lab in labels
        if "gender" in lab
        if lab.startswith("0:")
    ]

    LEGEND_FONTSIZE = 13
    LEGEND_TITLE_FONTSIZE = 14

    age_leg = top_ax.legend(
        age_handles,
        age_labels,
        bbox_to_anchor=legend_bboxes[0],
        loc="upper right",
        title="Age",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        # bbox_transform=fig.get_transform(),
    )
    eth_leg = top_ax.legend(
        ethnicity_handles,
        ethnicity_labels,
        bbox_to_anchor=legend_bboxes[1],
        loc="center right",
        title="Ethnicity",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        # bbox_transform=fig.get_transform(),
    )
    _gen_leg = top_ax.legend(
        gender_handles,
        gender_labels,
        bbox_to_anchor=legend_bboxes[2],
        loc="lower right",
        title="Gender",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        # bbox_transform=fig.get_transform(),
    )
    top_ax.add_artist(age_leg)
    top_ax.add_artist(eth_leg)

    # age_leg.set_in_layout(False)
    # eth_leg.set_in_layout(False)
    # _gen_leg.set_in_layout(False)

    def adjust_text_on_ax(texts: List[List[pltText]], ax):
        for t in texts:
            adjust_text(
                t,
                ax=ax,
                max_move=(0, 50),
                avoid_self=False,
                # force_explode=(0, 0.5),
                # force_text=(0, 0.2),
            )

    adjust_text_on_ax(top_texts, top_ax)
    adjust_text_on_ax(bot_texts, bot_ax)

    fig.tight_layout()

    trn = top_ax.transAxes.inverted()
    age_win = age_leg.get_window_extent().transformed(trn)
    eth_win = eth_leg.get_window_extent().transformed(trn)
    gen_win = _gen_leg.get_window_extent().transformed(trn)
    print(f"GOAL {(age_win.y0 + gen_win.y1) / 2}")
    print(f"ETH {(eth_win.y0 + eth_win.y1) / 2}")
    print(f"AGE->ETH = {age_win.y0 - eth_win.y1}")
    print(f"ETH->GEN = {eth_win.y0 - gen_win.y1}")

    # Show the plot
    if save_fig != "":
        fig.savefig(save_fig, dpi=600)

    return fig


def export_plot_data_to_csv(orig_data_fp, collected_data_fp, category_values, csv_fp):
    with open(orig_data_fp) as f1, open(collected_data_fp) as f2:
        orig_data = json.load(f1)
        coll_data = json.load(f2)

    emotions = RAF_DB_EMOTIONS
    biases = ["age", "ethnicity", "gender"]

    with open(csv_fp, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["Emotion", "Bias", "Category", "Original", "Collected"]
        writer.writerow(header)

        for em_ind, emotion in enumerate(emotions, start=1):
            orig_em = agg_values_to_pcts(orig_data[str(em_ind)]["categories"])
            coll_em = agg_values_to_pcts(coll_data[str(em_ind)]["categories"])
            for bias in biases:
                for cat in category_values[bias]:
                    orig_val = orig_em[bias].get(cat.lower(), 0)
                    coll_val = coll_em[bias].get(cat.lower(), 0)
                    writer.writerow([emotion, bias, cat, orig_val, coll_val])


if __name__ == "__main__":
    right = 1.06
    base = 0.05 / 0.99
    base *= 1.5
    plot(
        orig_data_fp="raf_utils/raf_agg.json",
        collected_data_fp="collected_raf/whole_raf_agg.json",
        save_fig="eval/graphs/side_by_side.pdf",
        category_values={
            "age": AGES,
            "ethnicity": sorted(RACES),
            "gender": sorted(GENDERS),
        },
        title="",
        in_colour=True,
        legend_bboxes=[(right, 0.9), (right, 0.2881272463715121), (right, -0.2)],
        use_hatch=False,
    )

    export_plot_data_to_csv(
        orig_data_fp="raf_utils/raf_agg.json",
        collected_data_fp="collected_raf/whole_raf_agg.json",
        category_values={
            "age": AGES,
            "ethnicity": sorted(RACES),
            "gender": sorted(GENDERS),
        },
        csv_fp="eval/graph_data/side_by_side_data.csv",
    )

    # plt.show()
