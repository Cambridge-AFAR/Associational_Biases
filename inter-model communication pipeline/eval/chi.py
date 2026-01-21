import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

from raf_utils import AGES, GENDERS, RACES
from utils import RAF_DB_EMOTIONS
from utils.phase import (
    PHASE_AGES,
    PHASE_GENDERS,
    PHASE_SKIN_TONES,
)


def raf_chi():
    orig_path = Path("raf_utils/raf_agg.json")
    coll_path = Path("collected_raf/whole_raf_agg.json")

    with orig_path.open() as f:
        orig = json.load(f)

    with coll_path.open() as f:
        coll = json.load(f)

    bias_vals = {
        "age": sorted(list(map(lambda x: x.lower(), AGES))),
        "ethnicity": sorted(list(map(lambda x: x.lower(), RACES))),
        "gender": sorted(list(map(lambda x: x.lower(), GENDERS))),
    }

    for emotion_ind in orig:
        emotion = RAF_DB_EMOTIONS[int(emotion_ind) - 1]
        print(f"{emotion.title()} Chi Square")
        for bias_type in orig[emotion_ind]["categories"]:
            orig_bias_dict = orig[emotion_ind]["categories"][bias_type]
            coll_bias_dict = coll[emotion_ind]["categories"][bias_type]
            vals = []
            vals.append([orig_bias_dict.get(b, 0) for b in bias_vals[bias_type]])
            vals.append([coll_bias_dict.get(b, 0) for b in bias_vals[bias_type]])

            data = np.array(vals)

            chi2, p, dof, _ = chi2_contingency(data)
            g, pg, _, _ = chi2_contingency(data, lambda_="log-likelihood")

            print(f"{bias_type}: chi={chi2:.4f}, dof={dof}, p={p:.4e}")
            if p < 0.001:
                p_print = r"$\mathbf{p} < 0.001^{***}$"
            elif p < 0.01:
                p_print = r"$\mathbf{p} < 0.01^{**}$"
            elif p < 0.05:
                p_print = r"$\mathbf{p} < 0.05^{*}$"
            else:
                p_print = r"$\mathbf{p} \geq 0.05^{}$"
            print(rf"$\chi^2({dof}) = {chi2:.1f}$,\ \ {p_print}")
            # print(f"{bias_type}: g={g:.4f}, p={pg:.4e}")
        print()


def phase_chi(task: str):
    short_task = f"{task[:3]}s"
    orig_path = Path(f"phase_anno/{task}_agg.json")
    coll_path = Path(f"phase_{short_task}/agg.json")

    with orig_path.open() as f:
        orig = json.load(f)

    with coll_path.open() as f:
        coll = json.load(f)

    bias_vals = {
        "age": sorted(
            [
                x.lower()
                for x in PHASE_AGES
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
        "skin tone": sorted(
            [
                x.lower()
                for x in PHASE_SKIN_TONES
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
        "gender": sorted(
            [
                x.lower()
                for x in PHASE_GENDERS
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
    }

    for category in orig:
        if category[0] == "_":
            continue
        print(category.title())
        for bias_type in ["age", "skin tone", "gender"]:
            orig_bias_dict = orig[category][bias_type]
            coll_bias_dict = coll[category][bias_type]

            vals = []
            vals.append([orig_bias_dict.get(b, 0) for b in bias_vals[bias_type]])
            vals.append([coll_bias_dict.get(b, 0) for b in bias_vals[bias_type]])

            data = np.array(vals)
            data = data[:, data.sum(axis=0) != 0]

            chi2, p, dof, _ = chi2_contingency(data)
            g, pg, _, _ = chi2_contingency(data, lambda_="log-likelihood")

            print(f"{bias_type}: chi={chi2:.4f}, dof={dof}, p={p:.4e}")
            if p < 0.001:
                p_print = r"$\mathbf{p} < 0.001^{***}$"
            elif p < 0.01:
                p_print = r"$\mathbf{p} < 0.01^{**}$"
            elif p < 0.05:
                p_print = r"$\mathbf{p} < 0.05^{*}$"
            else:
                p_print = r"$\mathbf{p} \geq 0.05^{\circ}$"
            print(rf"$\chi^2({dof}) = {chi2:.1f}$,\ \ {p_print}")
            # print(f"{bias_type}: g={g:.4f}, p={pg:.4e}")
        print()


if __name__ == "__main__":
    print("RAF")
    raf_chi()
    print("Phase emotions")
    phase_chi("emotion")
    print("Phase activities")
    phase_chi("activity")
