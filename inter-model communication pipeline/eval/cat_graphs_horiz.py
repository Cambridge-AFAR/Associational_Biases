import json

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from cycler import cycler

from raf_utils import AGES, GENDERS, RACES
from utils import RAF_DB_EMOTIONS
from utils.bias_categories import (
    AGE_CATEGORIES,
    ETHNICITY_CATEGORIES,
    GENDER_CATEGORIES,
)

colors = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

plt.rc("axes", prop_cycle=cycler(color=colors))


def category_sort(categories):
    cat_l = sorted(categories)
    cat_l.remove("other/unclear")
    cat_l.append("other/unclear")
    return cat_l


AGE_CATEGORIES = category_sort(AGE_CATEGORIES)
ETHNICITY_CATEGORIES = category_sort(ETHNICITY_CATEGORIES)
GENDER_CATEGORIES = category_sort(GENDER_CATEGORIES)


def get_category_values(category_data, categories_list):
    return [category_data.get(key, 0) for key in categories_list]


def plot(
    data_fp,
    title,
    save_fig="",
    category_values=(AGE_CATEGORIES, ETHNICITY_CATEGORIES, GENDER_CATEGORIES),
    xlim=-1,
):
    age_categories, ethnicity_categories, gender_categories = category_values
    with open(data_fp) as f:
        data = json.load(f)

    # Aggregate values for each category
    age_values = [
        get_category_values(data[str(i)]["categories"]["age"], age_categories)
        for i in range(1, 8)
    ]
    ethnicity_values = [
        get_category_values(
            data[str(i)]["categories"]["ethnicity"], ethnicity_categories
        )
        for i in range(1, 8)
    ]
    gender_values = [
        get_category_values(data[str(i)]["categories"]["gender"], gender_categories)
        for i in range(1, 8)
    ]

    # Combine for stacked bars
    values = {
        "Age": np.array(age_values).T,
        "Ethnicity": np.array(ethnicity_values).T,
        "Gender": np.array(gender_values).T,
    }

    # print(values)
    width = 0.8

    y = (np.arange(len(RAF_DB_EMOTIONS))) * 3 + (
        width / 2
    )  # Offset to align with bar centers

    fig, ax = plt.subplots(figsize=(14, 14))

    # hatches = ["//", "\\", "||", "++", "xx", "oo", ".."]

    max_height = 0
    for cat_name in values:
        max_height = max(max_height, np.max(np.sum(values[cat_name], axis=0)))

    ax.set_xlim(0, max_height * 1.05 if xlim == -1 else xlim)

    def plot_bias_category(cat_name, categories, offset=0.0):
        texts = []
        n_categories = len(categories)
        individual_height = width / n_categories  # Divide total height among categories
        for i, group in enumerate(categories):
            # Calculate individual offsets for each category within the group
            bar_offsets = y + offset + (i - n_categories // 2) * individual_height
            bars = ax.barh(
                bar_offsets,
                values[cat_name][i],
                individual_height,
                label=f"{cat_name}: {group}",
            )
            # Add category labels and size above each bar
            for bar, val in zip(bars, values[cat_name][i]):
                # if val == 0:
                #     continue
                label_x = (
                    bar.get_width() + 0.001 * max_height
                )  # Slight offset from the bar's end
                label_y = bar.get_y() + bar.get_height() / 2
                # Category label with size
                t = ax.text(
                    label_x,
                    label_y,
                    f"{group}: {val}",
                    ha="left",
                    va="center",
                    fontsize=8 * (1 - 1 / n_categories),
                )
                texts.append(t)

    for cat_name, cats, offset in zip(
        ["Age", "Ethnicity", "Gender"],
        [
            age_categories,
            ethnicity_categories,
            gender_categories,
        ],
        [-1.1 * width, 0.0, 1.1 * width],
    ):
        plot_bias_category(cat_name, cats, offset)

    # Customize plot
    ax.set_ylabel("Emotions")
    ax.set_yticks(y)
    # plt.tick_params(
    #     axis="x",  # changes apply to the x-axis
    #     which="both",  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=True,
    # )
    ax.set_yticklabels(RAF_DB_EMOTIONS)
    # for tl in ax.xaxis.get_majorticklabels():
    #     tl.set_y(tl.get_position()[1] - 0.01)

    ax.set_ylabel("Emotions")
    ax.set_xlabel("Counts")

    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()

    age_handles = [h for h, lab in zip(handles, labels) if lab.startswith("Age")]
    age_labels = [lab.split(":")[1].strip() for lab in labels if lab.startswith("Age")]
    ethnicity_handles = [
        h for h, lab in zip(handles, labels) if lab.startswith("Ethnicity")
    ]
    ethnicity_labels = [
        lab.split(":")[1].strip() for lab in labels if lab.startswith("Ethnicity")
    ]
    gender_handles = [h for h, lab in zip(handles, labels) if lab.startswith("Gender")]
    gender_labels = [
        lab.split(":")[1].strip() for lab in labels if lab.startswith("Gender")
    ]

    age_leg = ax.legend(
        age_handles,
        age_labels,
        bbox_to_anchor=(0.99, 1),
        loc="upper right",
        title="Age",
        fontsize="small",
    )
    eth_leg = plt.legend(
        ethnicity_handles,
        ethnicity_labels,
        bbox_to_anchor=(0.99, 0.9),
        loc="upper right",
        title="Ethnicity",
        fontsize="small",
    )
    _gen_leg = plt.legend(
        gender_handles,
        gender_labels,
        bbox_to_anchor=(0.99, 0.8),
        loc="upper right",
        title="Gender",
        fontsize="small",
    )
    ax.add_artist(age_leg)
    ax.add_artist(eth_leg)
    fig.tight_layout()

    # Show the plot
    if save_fig != "":
        fig.savefig(save_fig, dpi=600)

    return fig


if __name__ == "__main__":
    f = plot(
        "collected_raf_train/aggregated_results.json",
        title="Bias Categories by Emotion - Whole RAF",
        save_fig="eval/graphs/whole_raf_horiz.png",
        xlim=1200,
    )
    plot(
        "img_results/iters/hand_annotation.json",
        title="Bias Categories by Emotion - 5 Iterations",
        save_fig="eval/graphs/5_iters_horiz.png",
        xlim=6,
    )
    plot(
        "raf_utils/aggregated_metadata.json",
        title="Bias Categories by Emotion - Original RAF distribution",
        save_fig="eval/graphs/raf_original_horiz.png",
        category_values=(AGES, RACES, GENDERS),
        xlim=1000,
    )

    # plt.close(f)

    # plt.show()
