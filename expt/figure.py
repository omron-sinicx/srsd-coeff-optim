import os

import matplotlib.pyplot as plt
import numpy as np
from analyze import count_result_by_num_coeff, read_result
from path import PLOT_DIR, RESULT_DIR


def plots_by_num_coeff():
    other_min = 0
    other_max = 4
    expo_min = 0
    expo_max = 6
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * (other_max - other_min) + 7, expo_max - expo_min + 3.75)
    )
    files = [
        "n=20_none_lm_iter=1_beam=200_uniform_seed=0.json",
        "n=20_brute-force_lm-jump_iter=2_beam=10_uniform_seed=0.json",
        "n=20_brute-force_lm-jump_iter=2_beam=10_order_seed=0.json",
    ]
    titles = ["LM", "Ours with LM", "Ours with LM (estimated init)"]
    for i in range(3):
        data = read_result(os.path.join(RESULT_DIR, "srsd", files[i]))
        count = count_result_by_num_coeff(data)
        keys = [key for key in count.keys() if key != "total"]
        stats = np.zeros((expo_max - expo_min + 1, other_max - other_min + 1, 3))
        for key in keys:
            stats[key[0] - expo_min, key[1] - other_min, 0] = (
                count[key]["global optimum"] if "global optimum" in count[key] else 0
            )
            stats[key[0] - expo_min, key[1] - other_min, 1] = (
                count[key]["other failed"] if "other failed" in count[key] else 0
            ) + (count[key]["expo failed"] if "expo failed" in count[key] else 0)
            stats[key[0] - expo_min, key[1] - other_min, 2] = (
                count[key]["error"] if "error" in count[key] else 0
            )
        for x in range(expo_max - expo_min + 1):
            for y in range(other_max - other_min + 1):
                if np.sum(stats[x, y]) == 0:
                    continue
                axes[i].pie(
                    stats[x, y] / np.sum(stats[x, y]),
                    radius=0.5,
                    center=(y + 1, x + 1),
                    wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
                    colors=[
                        "#3cb44b",
                        "#4363d8",
                        "#e6194b",
                    ],  # https://sashamaps.net/docs/resources/20-colors/
                    hatch=["/", ".", "O"],
                )
        axes[i].set_yticks(np.arange(expo_max - expo_min + 1) + 1)
        axes[i].set_yticklabels(np.arange(expo_min, expo_max + 1), fontsize=20)
        axes[i].set_xticks(np.arange(other_max - other_min + 1) + 1)
        axes[i].set_xticklabels(np.arange(other_min, other_max + 1), fontsize=20)
        axes[i].grid(True, which="both", linestyle="--", linewidth=0.5)

        axes[i].set_ylim(0, expo_max - expo_min + 2)
        axes[i].set_xlim(0, other_max - other_min + 2)
        if i == 0:
            axes[i].set_ylabel(
                r"Number of exponential coefficients $\mathbf{c}_\mathrm{exp}$",
                fontsize=30,
            )
        if i == 1:
            axes[i].set_xlabel(
                r"Number of non-exponential coefficients $\mathbf{c}_\mathrm{non}$",
                fontsize=30,
            )
        axes[i].set_title(titles[i], fontsize=30)
    fig.legend(
        ["Global optimum", "Local optimum", "Failure"],
        loc="lower center",
        ncol=3,
        fontsize=30,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=1.1)
    plt.savefig(os.path.join(PLOT_DIR, "success_rate.pdf"))


def plots_by_num_coeff_poster():
    other_min = 0
    other_max = 4
    expo_min = 0
    expo_max = 6
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * (other_max - other_min) + 5, expo_max - expo_min + 3.75)
    )
    files = [
        "n=20_none_lm_iter=1_beam=200_uniform_seed=0.json",
        "n=20_brute-force_lm-jump_iter=2_beam=10_uniform_seed=0.json",
    ]
    titles = ["LM", "Ours with LM"]
    for i in range(2):
        data = read_result(os.path.join(RESULT_DIR, "srsd", files[i]))
        count = count_result_by_num_coeff(data)
        keys = [key for key in count.keys() if key != "total"]
        stats = np.zeros((expo_max - expo_min + 1, other_max - other_min + 1, 3))
        for key in keys:
            stats[key[0] - expo_min, key[1] - other_min, 0] = (
                count[key]["global optimum"] if "global optimum" in count[key] else 0
            )
            stats[key[0] - expo_min, key[1] - other_min, 1] = (
                count[key]["other failed"] if "other failed" in count[key] else 0
            ) + (count[key]["expo failed"] if "expo failed" in count[key] else 0)
            stats[key[0] - expo_min, key[1] - other_min, 2] = (
                count[key]["error"] if "error" in count[key] else 0
            )
        for x in range(expo_max - expo_min + 1):
            for y in range(other_max - other_min + 1):
                if np.sum(stats[x, y]) == 0:
                    continue
                axes[i].pie(
                    stats[x, y] / np.sum(stats[x, y]),
                    radius=0.5,
                    center=(y + 1, x + 1),
                    wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
                    colors=[
                        "#3cb44b",
                        "#4363d8",
                        "#e6194b",
                    ],  # https://sashamaps.net/docs/resources/20-colors/
                    hatch=["/", ".", "O"],
                )
        axes[i].set_yticks(np.arange(expo_max - expo_min + 1) + 1)
        axes[i].set_yticklabels(np.arange(expo_min, expo_max + 1), fontsize=20)
        axes[i].set_xticks(np.arange(other_max - other_min + 1) + 1)
        axes[i].set_xticklabels(np.arange(other_min, other_max + 1), fontsize=20)
        axes[i].grid(True, which="both", linestyle="--", linewidth=0.5)

        axes[i].set_ylim(0, expo_max - expo_min + 2)
        axes[i].set_xlim(0, other_max - other_min + 2)
        axes[i].set_ylabel(
            r"len($\mathbf{c}_\mathrm{exp}$)",
            fontsize=30,
        )
        axes[i].set_xlabel(
            r"len($\mathbf{c}_\mathrm{non}$)",
            fontsize=30,
        )
        axes[i].set_title(titles[i], fontsize=30)
    fig.legend(
        ["Global optimum", "Local optimum", "Failure"],
        loc="lower center",
        ncol=3,
        fontsize=30,
        bbox_to_anchor=(0.5, -0.01),
    )
    plt.subplots_adjust(bottom=0.18, top=0.95, left=0.07, right=0.97)
    plt.savefig(os.path.join(PLOT_DIR, "success_rate_poster.pdf"))


def plots_time_complexity():
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))

    # beamsize
    bs = [5, 10, 15, 20]
    times = [
        9.684711895883083 * 2,
        18.986270486244134 * 2,
        28.028972495879444 * 2,
        36.90880356303283 * 2,
    ]
    exp_times = [
        6.6819977675165445 * 2,
        12.31998930552176 * 2,
        18.163629482899392 * 2,
        23.748056638453686 * 2,
    ]
    nonexp_times = [
        3.0027141283665384 * 2,
        6.666281180722373 * 2,
        9.865343012980052 * 2,
        13.160746924579144 * 2,
    ]
    mrs = [47, 64, 65, 65]
    mrs = [mr * 100 / 112 for mr in mrs]
    axes[0].plot(bs, times, label="Total time [s]", color="#f58231", marker="o")
    axes[0].plot(
        bs,
        exp_times,
        label=r"Time for optimizing $\mathbf{c}_\mathrm{exp}$ [s]",
        color="#e6194B",
        marker="o",
    )
    axes[0].plot(
        bs,
        nonexp_times,
        label=r"Time for optimizing $\mathbf{c}_\mathrm{non}$ [s]",
        color="#4363d8",
        marker="o",
    )
    axes_0 = axes[0].twinx()
    axes_0.bar(bs, mrs, color="#3cb44b", alpha=0.3, label="Success rate [\%]", width=2)
    axes[0].set_ylim(0, 90)
    axes[0].set_xlabel(r"Beam size $B$")
    axes[0].set_ylabel("Time [s]")
    axes_0.set_ylim(0, 100)
    axes_0.set_xticks(bs)
    axes_0.set_ylabel("Success rate [\%]")

    # outiter
    iters = [1, 2, 3, 4]
    times = [
        11.056320011615753 * 1,
        18.986270486244134 * 2,
        21.067612870108512 * 3,
        21.419914804399014 * 4,
    ]
    exp_times = [
        2.7116866963250295 * 1,
        12.31998930552176 * 2,
        15.356295877269336 * 3,
        16.459845679146902 * 4,
    ]
    nonexp_times = [
        8.344633315290723 * 1,
        6.666281180722373 * 2,
        5.711316992839177 * 3,
        4.960069125252111 * 4,
    ]
    mrs = [53, 64, 64, 64]
    mrs = [mr * 100 / 112 for mr in mrs]
    axes[1].plot(iters, times, color="#f58231", marker="o")
    axes[1].plot(
        iters,
        exp_times,
        color="#e6194B",
        marker="o",
    )
    axes[1].plot(
        iters,
        nonexp_times,
        color="#4363d8",
        marker="o",
    )
    axes_1 = axes[1].twinx()
    axes_1.bar(iters, mrs, color="#3cb44b", alpha=0.3, width=0.4)
    axes[1].set_ylim(0, 90)
    axes[1].set_xlabel(r"Number of outer iterations $L_\mathrm{out}$")
    axes[1].set_ylabel("Time [s]")
    axes_1.set_ylim(0, 100)
    axes_1.set_xticks(iters)
    axes_1.set_ylabel("Success rate [\%]")

    # n
    ns = [10, 20, 30, 40]
    times = [
        10.942976862192154 * 2,
        18.986270486244134 * 2,
        26.597852719681605 * 2,
        36.50028135095324 * 2,
    ]
    exp_times = [
        7.430542517985616 * 2,
        12.31998930552176 * 2,
        15.687648713588715 * 2,
        21.34254672058991 * 2,
    ]
    nonexp_times = [
        3.5124343442065373 * 2,
        6.666281180722373 * 2,
        10.910204006092888 * 2,
        15.157734630363327 * 2,
    ]
    mrs = [53, 64, 64, 64]
    mrs = [mr * 100 / 112 for mr in mrs]
    axes[2].plot(ns, times, color="#f58231", marker="o")
    axes[2].plot(
        ns,
        exp_times,
        color="#e6194B",
        marker="o",
    )
    axes[2].plot(
        ns,
        nonexp_times,
        color="#4363d8",
        marker="o",
    )
    axes_2 = axes[2].twinx()
    axes_2.bar(ns, mrs, color="#3cb44b", alpha=0.3, width=4)
    axes[2].set_ylim(0, 90)
    axes[2].set_xlabel(r"Data size $n$")
    axes[2].set_ylabel("Time [s]")
    axes_2.set_ylim(0, 100)
    axes_2.set_xticks(ns)
    axes_2.set_ylabel("Success rate [\%]")
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.05, right=0.95, wspace=0.5)

    # shared legend
    lines, labels = axes[0].get_legend_handles_labels()
    lines2, labels2 = axes_0.get_legend_handles_labels()
    fig.legend(
        lines + lines2,
        labels + labels2,
        loc="lower center",
        ncol=4,
    )
    plt.savefig(os.path.join(PLOT_DIR, "time_complexity.pdf"))


def plots_time_complexity_poster():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # beamsize
    bs = [5, 10, 15, 20]
    times = [
        9.684711895883083 * 2,
        18.986270486244134 * 2,
        28.028972495879444 * 2,
        36.90880356303283 * 2,
    ]
    exp_times = [
        6.6819977675165445 * 2,
        12.31998930552176 * 2,
        18.163629482899392 * 2,
        23.748056638453686 * 2,
    ]
    nonexp_times = [
        3.0027141283665384 * 2,
        6.666281180722373 * 2,
        9.865343012980052 * 2,
        13.160746924579144 * 2,
    ]
    mrs = [47, 64, 65, 65]
    mrs = [mr * 100 / 112 for mr in mrs]
    axes[0][0].plot(bs, times, label="Total time [s]", color="#f58231", marker="o")
    axes[0][0].plot(
        bs,
        exp_times,
        label=r"Time for optimizing $\mathbf{c}_\mathrm{exp}$ [s]",
        color="#e6194B",
        marker="o",
    )
    axes[0][0].plot(
        bs,
        nonexp_times,
        label=r"Time for optimizing $\mathbf{c}_\mathrm{non}$ [s]",
        color="#4363d8",
        marker="o",
    )
    axes_00 = axes[0][0].twinx()
    axes_00.bar(bs, mrs, color="#3cb44b", alpha=0.3, label="Success rate [\%]", width=2)
    axes[0][0].set_ylim(0, 90)
    axes[0][0].set_xlabel(r"Beam size $B$")
    axes[0][0].set_ylabel("Time [s]")
    axes_00.set_ylim(0, 100)
    axes_00.set_xticks(bs)
    axes_00.set_ylabel("Success rate [\%]")

    # outiter
    iters = [1, 2, 3, 4]
    times = [
        11.056320011615753 * 1,
        18.986270486244134 * 2,
        21.067612870108512 * 3,
        21.419914804399014 * 4,
    ]
    exp_times = [
        2.7116866963250295 * 1,
        12.31998930552176 * 2,
        15.356295877269336 * 3,
        16.459845679146902 * 4,
    ]
    nonexp_times = [
        8.344633315290723 * 1,
        6.666281180722373 * 2,
        5.711316992839177 * 3,
        4.960069125252111 * 4,
    ]
    mrs = [53, 64, 64, 64]
    mrs = [mr * 100 / 112 for mr in mrs]
    axes[0][1].plot(iters, times, color="#f58231", marker="o")
    axes[0][1].plot(
        iters,
        exp_times,
        color="#e6194B",
        marker="o",
    )
    axes[0][1].plot(
        iters,
        nonexp_times,
        color="#4363d8",
        marker="o",
    )
    axes_01 = axes[0][1].twinx()
    axes_01.bar(iters, mrs, color="#3cb44b", alpha=0.3, width=0.4)
    axes[0][1].set_ylim(0, 90)
    axes[0][1].set_xlabel(r"Number of outer iterations $L_\mathrm{out}$")
    axes[0][1].set_ylabel("Time [s]")
    axes_01.set_ylim(0, 100)
    axes_01.set_xticks(iters)
    axes_01.set_ylabel("Success rate [\%]")

    # n
    ns = [10, 20, 30, 40]
    times = [
        10.942976862192154 * 2,
        18.986270486244134 * 2,
        26.597852719681605 * 2,
        36.50028135095324 * 2,
    ]
    exp_times = [
        7.430542517985616 * 2,
        12.31998930552176 * 2,
        15.687648713588715 * 2,
        21.34254672058991 * 2,
    ]
    nonexp_times = [
        3.5124343442065373 * 2,
        6.666281180722373 * 2,
        10.910204006092888 * 2,
        15.157734630363327 * 2,
    ]
    mrs = [53, 64, 64, 64]
    mrs = [mr * 100 / 112 for mr in mrs]
    axes[1][0].plot(ns, times, color="#f58231", marker="o")
    axes[1][0].plot(
        ns,
        exp_times,
        color="#e6194B",
        marker="o",
    )
    axes[1][0].plot(
        ns,
        nonexp_times,
        color="#4363d8",
        marker="o",
    )
    axes_10 = axes[1][0].twinx()
    axes_10.bar(ns, mrs, color="#3cb44b", alpha=0.3, width=4)
    axes[1][0].set_ylim(0, 90)
    axes[1][0].set_xlabel(r"Data size $n$")
    axes[1][0].set_ylabel("Time [s]")
    axes_10.set_ylim(0, 100)
    axes_10.set_xticks(ns)
    axes_10.set_ylabel("Success rate [\%]")
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.05, right=0.95, wspace=0.5)

    # shared legend
    lines, labels = axes[0][0].get_legend_handles_labels()
    lines2, labels2 = axes_00.get_legend_handles_labels()
    axes[1][1].axis("off")
    axes[1][1].legend(
        lines + lines2,
        labels + labels2,
        loc="center",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "time_complexity_poster.pdf"))


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 30
    plots_by_num_coeff()
    plots_time_complexity()
    plots_by_num_coeff_poster()
    plots_time_complexity_poster()
