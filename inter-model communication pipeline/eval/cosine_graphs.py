import os
from itertools import cycle
from typing import Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from cycler import cycler

line_styles = [
    "-",
    "--",
    "-.",
]

colors = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

rc_fonts = {
    "text.usetex": True,
    "text.latex.preamble": "\n".join(
        [r"\usepackage{libertine}", r"\usepackage[libertine]{newtxmath}"]
    ),
}
mpl.rcParams.update(rc_fonts)

plt.rc("axes", prop_cycle=cycler(color=colors))


def plot_csv(
    csv_fp: str,
    output_fp: Union[None, str],
    y_lim: Tuple[float, float],
    title: str,
    pairwise: bool = False,
    add_text_labels: bool = False,
    legend_bbox: Tuple[float, float] = (1.01, 1),
    legend: bool = True,
    legend_loc: str = "upper left",
    legend_ncol: int = 1,
):
    if output_fp is not None:
        os.makedirs(os.path.split(output_fp)[0], exist_ok=True)
    else:
        output_fp = csv_fp.replace(".csv", ".png")

    df = pd.read_csv(csv_fp)

    if "label" in df.columns:
        transform_df(df)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.grid(axis="y")

    ax.set_yticks(np.arange(0.0, 1.01, 0.05))
    ax.set_ylim(*y_lim)

    AXES_FONT_SIZE = 38

    ax.set_ylabel("Cosine similarity", fontsize=AXES_FONT_SIZE)

    if pairwise:
        plt.xticks([0, 1, 2, 3], ["1-2", "2-3", "3-4", "4-5"])
        ax.set_xlabel("Iteration pair", fontsize=AXES_FONT_SIZE)

    texts = []
    for (name, *data), line_style in zip(
        df.itertuples(index=False), cycle(line_styles)
    ):
        (line,) = ax.plot(data, linestyle=line_style, label=name, marker="o")

        for i, value in enumerate(data):
            if add_text_labels:
                t = ax.text(
                    i,
                    value,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=line.get_color(),
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        boxstyle="round,pad=0",
                        alpha=0.5,
                    ),
                )
                texts.append(t)

    TICK_LABEL_SIZE = 26
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    if add_text_labels:
        adjust_text(
            texts,
            ax=ax,
            only_move={"points": "xy"},
            max_move=15,
            # arrowprops=dict(arrowstyle="->", color='black', lw=0.5)
        )  # type: ignore

    LEGEND_LABEL_SIZE = 18
    if legend:
        ax.legend(
            bbox_to_anchor=legend_bbox,
            loc=legend_loc,
            borderaxespad=0.0,
            fontsize=LEGEND_LABEL_SIZE,
            ncol=legend_ncol,
        )

    if title != "":
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_fp, dpi=600)
    # plt.show()


def transform_df(df: pd.DataFrame):
    df.insert(
        0,
        "name",
        df.apply(
            lambda x: f"{x['label']} - {x['prompt_name'].split('_')[1]}.png", axis=1
        ),
    )
    df.drop(columns=["prompt_name", "label"], inplace=True)
    df.sort_values(by="name", inplace=True)


if __name__ == "__main__":
    raf_db_4 = [
        {
            "csv_fp": "similarities/raf-db-4/desc_to_desc_pairwise.csv",
            "output_fp": "eval/graphs/txt_desc_to_desc.pdf",
            # "title": "Pairwise description similarities - initial text generation",
            "title": "",
            "y_lim": (0.8, 1.0),
            "pairwise": True,
            "legend": False,
        },
        {
            "csv_fp": "similarities/raf-db-4/image_to_image_pairwise.csv",
            "output_fp": "eval/graphs/txt_img_to_img.pdf",
            # "title": "Pairwise image similarities - initial text generation",
            "title": "",
            "y_lim": (0.65, 1.0),
            "pairwise": True,
            "legend_bbox": (0.01, 0.01),
            "legend_loc": "lower left",
        },
    ]

    raf_from_images = [
        {
            "csv_fp": "similarities/raf-db-from-images/desc_to_desc_pairwise.csv",
            "output_fp": "eval/graphs/raf_desc_to_desc.pdf",
            # "title": "Pairwise description similarities - initial RAF-DB image",
            "title": "",
            "y_lim": (0.65, 1.0),
            "pairwise": True,
            "legend": False,
        },
        {
            "csv_fp": "similarities/raf-db-from-images/image_to_image_pairwise.csv",
            "output_fp": "eval/graphs/raf_img_to_img.pdf",
            # "title": "Pairwise image similarities - initial RAF-DB image",
            "title": "",
            "legend_bbox": (0.99, 0.01),
            "legend_loc": "lower right",
            "legend_ncol": 2,
            "y_lim": (0.5, 1.0),
            "pairwise": True,
        },
    ]

    for x in raf_db_4:
        plot_csv(**x)

    for y in raf_from_images:
        plot_csv(**y)
    # plot_csv(**raf_from_images[1])
