import csv
import json
from typing import Dict, List, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.text import Text as pltText

from eval.cat_graphs import (
    greyscale,
)
from utils.phase import (
    PHASE_ACTIVITIES,
    PHASE_AGES,
    PHASE_EMOTIONS,
    PHASE_GENDERS,
    PHASE_SKIN_TONES,
    shorten_label,
)

colourful = [
    # "#E69F00",  # orange
    "#0072B2",  # blue
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

BIASES = ["age", "skin tone", "gender"]

rc_fonts = {
    "text.usetex": True,
    "text.latex.preamble": "\n".join(
        [r"\usepackage{libertine}", r"\usepackage[libertine]{newtxmath}"]
    ),
}
mpl.rcParams.update(rc_fonts)


def agg_values_to_pcts(cat_dict):
    res = {
        bias: {
            cat: (val / sum(cat_dict[bias].values())) * 100
            for cat, val in cat_dict[bias].items()
        }
        for bias in BIASES
    }

    for bias in res.values():
        dis = bias.pop("disagreement", 0)
        bias["unsure"] = bias.get("unsure", 0) + dis

    return res


def plot(
    orig_data_fp,
    collected_data_fp,
    title,
    task: Literal["activites", "emotions"],
    save_fig="",
    category_values: Dict[str, List[str]] = {
        "age": PHASE_AGES,
        "skin tone": PHASE_SKIN_TONES,
        "gender": PHASE_GENDERS,
    },
    legend_bboxes=[(1.05, 1), (1.05, 0.8), (1.05, 0.6)],
    *,
    in_colour: bool = True,
    use_hatch: bool = False,
):
    with open(orig_data_fp) as f1, open(collected_data_fp) as f2:
        orig_data = json.load(f1)
        coll_data = json.load(f2)

    if task == "emotions":
        task_list = PHASE_EMOTIONS
    else:
        task_list = [
            x for x in PHASE_ACTIVITIES if x.lower() not in ("other", "unsure")
        ]

    orig_values_to_plot = {
        tv: agg_values_to_pcts(orig_data[shorten_label(tv)]) for tv in task_list
    }

    coll_values_to_plot = {
        tv: agg_values_to_pcts(coll_data[shorten_label(tv)]) for tv in task_list
    }

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

    # top will have 3 emotions, bottom will have 3 for emotions
    if task == "emotions":
        x_top = np.arange(3) * TICK_MULTIPLIER
        x_bot = np.arange(3) * TICK_MULTIPLIER
        ind_loc = 3
    else:
        x_top = np.arange(5) * TICK_MULTIPLIER
        x_bot = np.arange(4) * TICK_MULTIPLIER
        ind_loc = 5

    fig, (top_ax, bot_ax) = plt.subplots(2, 1, figsize=(16, 10))

    top_ax.set_ylim(0, 105)
    bot_ax.set_ylim(0, 105)

    top_texts: List[List[pltText]] = []
    bot_texts: List[List[pltText]] = []

    OFFSET = 1.1

    for task_ind, task_val in enumerate(task_list):
        origs = orig_values_to_plot[task_val]
        colls = coll_values_to_plot[task_val]

        for bias_type, offset in zip(BIASES, [-OFFSET, 0, OFFSET]):
            orig_ys = np.array(
                [
                    origs[bias_type].get(cat.lower(), 0)
                    for cat in category_values[bias_type]
                ]
            )
            coll_ys = np.array(
                [
                    colls[bias_type].get(cat.lower(), 0)
                    for cat in category_values[bias_type]
                ]
            )

            if task_ind < ind_loc:
                ratio = 0.48 / 0.35
                ax = top_ax
                texts = top_texts
                bar_loc = task_ind
                width = 0.4
                side_off = 0.2 * ratio
                offset *= 1.12
            else:
                ax = bot_ax
                texts = bot_texts
                bar_loc = task_ind - ind_loc
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
                        fontsize=14 if task == "emotions" else 10,
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
                    -2.5,
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
                    f"{task_ind}:{bias_type}:{cat}"
                    for cat in category_values[bias_type]
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
                #     f"{task_ind}:{bias_type}:{cat}" for cat in category_values[bias_type]
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
                -6,
                bias_type.title(),
                ha="center",
                va="center",
                fontsize=14,
                rotation=0,
            )

    def config_ax(ax: Axes, x: np.ndarray, sl: slice):
        ax.set_xticks(x)
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True,
            labelsize=16,
            pad=25,
        )
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xticklabels(map(lambda x: x.title(), task_list[sl]))
        # for tl in ax.xaxis.get_majorticklabels():
        #     tl.set_y(tl.get_position()[1] - 0.05)

        ax.set_ylabel(r"Proportions / \%", fontsize=18)

        if title != "":
            ax.set_title(title)

    config_ax(top_ax, x_top, slice(0, ind_loc))
    config_ax(bot_ax, x_bot, slice(ind_loc, None))

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
        if "skin tone" in lab
        if lab.startswith("0:")
    ]
    ethnicity_labels = [
        lab.split(":")[2].strip().title()
        for lab in labels
        if "skin tone" in lab
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
        title="Skin Tone",
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


def export_data_to_csv(
    orig_data_fp,
    collected_data_fp,
    out_csv_fp,
    task: Literal["activites", "emotions"],
    category_values: Dict[str, List[str]] = {
        "age": PHASE_AGES,
        "skin tone": PHASE_SKIN_TONES,
        "gender": PHASE_GENDERS,
    },
):
    with open(orig_data_fp) as f1, open(collected_data_fp) as f2:
        orig_data = json.load(f1)
        coll_data = json.load(f2)

    if task == "emotions":
        task_list = PHASE_EMOTIONS
    else:
        task_list = [
            x for x in PHASE_ACTIVITIES if x.lower() not in ("other", "unsure")
        ]

    orig_values_to_plot = {
        tv: agg_values_to_pcts(orig_data[shorten_label(tv)]) for tv in task_list
    }
    coll_values_to_plot = {
        tv: agg_values_to_pcts(coll_data[shorten_label(tv)]) for tv in task_list
    }

    with open(out_csv_fp, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["task_value", "bias", "bias_category", "original_pct", "collected_pct"]
        )
        for task_val in task_list:
            for bias_type in BIASES:
                for cat in category_values[bias_type]:
                    orig_pct = orig_values_to_plot[task_val][bias_type].get(
                        cat.lower(), 0
                    )
                    coll_pct = coll_values_to_plot[task_val][bias_type].get(
                        cat.lower(), 0
                    )
                    writer.writerow(
                        [task_val, bias_type, cat, f"{orig_pct:.2f}", f"{coll_pct:.2f}"]
                    )


if __name__ == "__main__":
    right = 1.06
    base = 0.05 / 0.99
    base *= 1.5
    plot(
        orig_data_fp="phase_anno/emotion_agg.json",
        collected_data_fp="phase_emos/agg.json",
        save_fig="eval/graphs/phase_emos.pdf",
        category_values={
            "age": PHASE_AGES,
            "skin tone": sorted(PHASE_SKIN_TONES),
            "gender": sorted(PHASE_GENDERS),
        },
        title="",
        task="emotions",
        in_colour=True,
        legend_bboxes=[(right, 0.9), (right, 0.2571908695572681), (right, -0.2)],
        use_hatch=False,
    )
    plot(
        orig_data_fp="phase_anno/activity_agg.json",
        collected_data_fp="phase_acts/agg.json",
        save_fig="eval/graphs/phase_acts.pdf",
        category_values={
            "age": PHASE_AGES,
            "skin tone": sorted(PHASE_SKIN_TONES),
            "gender": sorted(PHASE_GENDERS),
        },
        title="",
        task="activites",
        in_colour=True,
        legend_bboxes=[(right, 0.9), (right, 0.2571908695572681), (right, -0.2)],
        use_hatch=False,
    )

    export_data_to_csv(
        orig_data_fp="phase_anno/emotion_agg.json",
        collected_data_fp="phase_emos/agg.json",
        out_csv_fp="eval/graph_data/phase_emos.csv",
        task="emotions",
        category_values={
            "age": PHASE_AGES,
            "skin tone": sorted(PHASE_SKIN_TONES),
            "gender": sorted(PHASE_GENDERS),
        },
    )
    export_data_to_csv(
        orig_data_fp="phase_anno/activity_agg.json",
        collected_data_fp="phase_acts/agg.json",
        out_csv_fp="eval/graph_data/phase_acts.csv",
        task="activites",
        category_values={
            "age": PHASE_AGES,
            "skin tone": sorted(PHASE_SKIN_TONES),
            "gender": sorted(PHASE_GENDERS),
        },
    )

    # target 0.362

    # plt.show()
