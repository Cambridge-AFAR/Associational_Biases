import json
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from raf_utils import AGES, GENDERS, RACES
from utils import RAF_DB_EMOTIONS
from utils.bias_categories import (
    AGE_CATEGORIES,
    ETHNICITY_CATEGORIES,
    GENDER_CATEGORIES,
)

rc_fonts = {
    "text.usetex": True,
    "text.latex.preamble": "\n".join(
        [r"\usepackage{libertine}", r"\usepackage[libertine]{newtxmath}"]
    ),
}
mpl.rcParams.update(rc_fonts)

colourful = [
    # "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#F0E442",  # yellow
    # "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

# greyscale = [
#     "#000000",  # Black
#     "#222222",  # Very dark grey
#     "#444444",  # Dark grey
#     "#666666",  # Medium-dark grey
#     "#888888",  # Medium grey
#     "#AAAAAA",  # Light grey
#     "#CCCCCC",  # Lighter grey
#     "#EEEEEE",  # Very light grey
#     "#FFFFFF",  # White
# ]
greyscale = ["#d9d9d9", "#bdbdbd", "#969696", "#636363", "#252525"]  # Light to dark


def category_sort(categories):
    cat_l = sorted(categories)
    cat_l.remove("other/unclear")
    cat_l.append("other/unclear")
    return cat_l


def convert_raf_ethnicity(eth: str):
    eth = eth.lower()
    if eth == "caucasian":
        eth = "white"
    elif eth == "african-american":
        eth = "black"
    return eth.title()


AGE_CATEGORIES = category_sort(AGE_CATEGORIES)
ETHNICITY_CATEGORIES = category_sort(ETHNICITY_CATEGORIES)
GENDER_CATEGORIES = category_sort(GENDER_CATEGORIES)

BIASES = ["age", "ethnicity", "gender"]


def agg_values_to_pcts(categories_dict):
    return {
        bias: {
            cat: (val / categories_dict[bias]["_metadata"]["total"]) * 100
            for cat, val in categories_dict[bias].items()
            if cat != "_metadata"
        }
        for bias in BIASES
    }


def plot(
    data_fp,
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
    with open(data_fp) as f:
        data = json.load(f)

    values_to_plot = [
        agg_values_to_pcts(data[str(i)]["categories"]) for i in range(1, 8)
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
    x = np.arange(len(RAF_DB_EMOTIONS)) * 3  # Add space between bar groups
    width = 0.8  # Width of each bar within a group

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.set_ylim(0, 105)

    texts = []

    for em_ind, emotion in enumerate(values_to_plot, start=0):
        for bias_type, offset in zip(BIASES, [-1.2 * width, 0, 1.2 * width]):
            # print(f"{em_ind} - {bias_type}")
            ys = np.array(
                [
                    emotion[bias_type].get(cat.lower(), 0)
                    for cat in category_values[bias_type]
                ]
            )
            # print(ys)
            bars = ax.bar(
                em_ind * 3 + offset,
                ys,
                width=width,
                label=[
                    f"{em_ind}:{bias_type}:{cat}" for cat in category_values[bias_type]
                ],
                bottom=np.cumsum(np.insert(ys, 0, 0)[:-1]),
                color=[v for _, v in colour_map[bias_type].items()],
                hatch=[v for _, v in hatch_map[bias_type].items()]
                if use_hatch
                else None,
            )

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
                    fontsize=16,
                    bbox=dict(
                        facecolor=(1, 1, 1, 0.5),
                        edgecolor="none",
                        # pad=0.01,
                        boxstyle="round,pad=0.01,rounding_size=0.2",
                    ),
                )
                texts.append(t)
                ax.text(
                    label_x,
                    -5,
                    bias_type.title(),
                    ha="center",
                    va="center",
                    fontsize=15,
                    rotation=30,
                )

    # Customize plot
    ax.set_xlabel("Emotions", fontsize=20)
    ax.set_xticks(x)
    ax.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True,
        labelsize=18,
        pad=40,
    )
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xticklabels(RAF_DB_EMOTIONS)
    # for tl in ax.xaxis.get_majorticklabels():
    #     tl.set_y(tl.get_position()[1] - 0.05)

    ax.set_ylabel(r"Proportions / \%", fontsize=20)

    if title != "":
        ax.set_title(title)

    ax.legend()

    handles, labels = ax.get_legend_handles_labels()

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
        lab.split(":")[2].strip().title()
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

    age_leg = ax.legend(
        age_handles,
        age_labels,
        bbox_to_anchor=legend_bboxes[0],
        loc="upper right",
        title="Age",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )
    eth_leg = plt.legend(
        ethnicity_handles,
        ethnicity_labels,
        bbox_to_anchor=legend_bboxes[1],
        loc="center right",
        title="Ethnicity",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )
    _gen_leg = plt.legend(
        gender_handles,
        gender_labels,
        bbox_to_anchor=legend_bboxes[2],
        loc="upper right",
        title="Gender",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )
    ax.add_artist(age_leg)
    ax.add_artist(eth_leg)
    fig.tight_layout()

    trn = ax.transAxes.inverted()
    age_win = age_leg.get_window_extent().transformed(trn)
    eth_win = eth_leg.get_window_extent().transformed(trn)
    gen_win = _gen_leg.get_window_extent().transformed(trn)
    print(f"GOAL {(age_win.y0 + gen_win.y1) / 2}")
    print(f"ETH {(eth_win.y0 + eth_win.y1) / 2}")
    print(f"AGE->ETH = {age_win.y0 - eth_win.y1}")
    print(f"ETH->GEN = {eth_win.y0 - gen_win.y1}")

    adjust_text(
        texts,
        ax=ax,
        max_move=(0, 10),
        avoid_self=False,
    )

    # Show the plot
    if save_fig != "":
        fig.savefig(save_fig, dpi=600)

    return fig


if __name__ == "__main__":
    right = 1.1
    # f = plot(
    #     "collected_raf_train/aggregated_results.json",
    #     # title="Bias Categories by Emotion - Whole RAF",
    #     title="",
    #     save_fig="eval/graphs/whole_raf.png",
    #     legend_bboxes=[(right, 0.95), (right, 0.725), (right, 0.5)],
    # )
    plot(
        "img_results/iters/llava_agg.json",
        # title="Bias Categories by Emotion - 5 Iterations",
        title="",
        in_colour=True,
        save_fig="eval/graphs/5_iters.pdf",
        category_values={
            "age": AGES,
            "ethnicity": sorted(RACES),
            "gender": sorted(GENDERS),
        },
        legend_bboxes=[(right, 0.95), (right, 0.4410487115178265), (right, 0.3)],
        use_hatch=False,
    )
    # plot(
    #     "raf_utils/aggregated_metadata.json",
    #     # title="Bias Categories by Emotion - Original RAF distribution",
    #     title="",
    #     save_fig="eval/graphs/raf_original.png",
    #     category_values={
    #         "age": sorted(AGES),
    #         "ethnicity": sorted(RACES),
    #         "gender": sorted(GENDERS),
    #     },
    #     legend_bboxes=[(right, 0.95), (right - 0.025, 0.725), (right, 0.55)],
    # )
    # plot(
    #     "collected_raf/whole_raf_agg.json",
    #     category_values={
    #         "age": AGES,
    #         "ethnicity": sorted(RACES),
    #         "gender": sorted(GENDERS),
    #     },
    #     title="",
    #     save_fig="eval/graphs/whole_collected_raf.png",
    #     legend_bboxes=[(right, 0.95), (right - 0.025, 0.725), (right, 0.55)],
    # )
    # plot(
    #     "raf_utils/raf_agg.json",
    #     category_values={
    #         "age": AGES,
    #         "ethnicity": sorted(RACES),
    #         "gender": sorted(GENDERS),
    #     },
    #     title="",
    #     save_fig="eval/graphs/whole_orig_raf.png",
    #     legend_bboxes=[(right, 0.95), (right - 0.025, 0.725), (right, 0.55)],
    # )

    # plt.close(f)

    plt.show()
