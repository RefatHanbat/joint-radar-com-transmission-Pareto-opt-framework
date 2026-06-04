import numpy as np

import matplotlib.pyplot as plt


def myf_plot_fig3(results):

    curve_styles = {
        "Radar-only": {"color": "red", "linestyle": "-", "label": "Radar-only"},
        "Joint-K = 1": {"color": "blue", "linestyle": "--", "label": r"Joint-$K=1$"},
        "Joint-K = 2": {"color": "black", "linestyle": "-.", "label": r"Joint-$K=2$"},
        "Joint-K = 3": {"color": "red", "linestyle": ":", "label": r"Joint-$K=3$"},
    }

    for N in results.keys():

        plt.figure(figsize=(8, 5))

        for curve_name, style in curve_styles.items():

            if curve_name not in results[N]:
                continue

            theta = results[N][curve_name]["theta"]
            pattern_dB = results[N][curve_name]["pattern_dB"]

            plt.plot(
                theta,
                pattern_dB,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                label=style["label"],
            )

        plt.xlabel(r"$\theta$ (deg)")
        plt.ylabel("Beampattern (dBi)")
        plt.title(
            rf"The optimized beampatterns, $N={N}$, "
            rf"$\Gamma_k=20$ dB, $K=0,1,2,3$"
        )

        plt.grid(True)
        plt.legend()

        plt.xlim([-90, 90])
        plt.ylim([-70, 20])

        plt.tight_layout()

        plt.savefig(f"fig3_N{N}.png", dpi=300)

        plt.show()


def myf_plot_fig4_all_methods(results_fig4):

    plt.figure(figsize=(10, 6))

    # ==================================================
    # Radar-only benchmark
    # ==================================================

    if "Radar-only" in results_fig4:

        N_radar = results_fig4["Radar-only"]["N"]

        eta_radar_dB = results_fig4["Radar-only"]["eta_avg_dB"]

        plt.plot(
            N_radar,
            eta_radar_dB,
            color="red",
            linestyle="-",
            marker="*",
            linewidth=2.8,
            markersize=11,
            label="Radar-only"
        )

    # ==================================================
    # Joint schemes
    # ==================================================

    for K in results_fig4.keys():

        if K == "Radar-only":

            continue

        N = results_fig4[K]["N"]

        eta_proposed_dB = results_fig4[K]["eta_proposed_avg_dB"]

        eta_similarity_dB = results_fig4[K]["eta_similarity_avg_dB"]

        eta_waveform22_dB = results_fig4[K]["eta_waveform22_avg_dB"]

        # Proposed method
        plt.plot(
            N,
            eta_proposed_dB,
            marker="o",
            linestyle="-",
            linewidth=2,
            label=rf"Proposed, $K={K}$"
        )

        # Paper [21] covariance similarity
        plt.plot(
            N,
            eta_similarity_dB,
            marker="s",
            linestyle="--",
            linewidth=2,
            label=rf"Similarity P21, $K={K}$"
        )

        # Paper [22] waveform similarity
        plt.plot(
            N,
            eta_waveform22_dB,
            marker="^",
            linestyle=":",
            linewidth=2,
            label=rf"Waveform P22, $K={K}$"
        )

    plt.xlabel("Number of antennas, N")

    plt.ylabel("Average DPSL (dB)")

    plt.title("Fig. 4: Radar-only, Proposed, P21 and P22 Benchmarks")

    plt.grid(True)

    plt.legend(fontsize=9)

    plt.tight_layout()

    plt.savefig("fig4_all_methods_with_radar_only.png", dpi=300)

    plt.show()


def myf_plot_fig5_all_methods(results_fig5, N):

    plt.figure(figsize=(10, 6))

    # ==================================================
    # Radar-only benchmark
    # ==================================================

    if "Radar-only" in results_fig5:

        Gamma_radar = results_fig5["Radar-only"]["Gamma_dB"]

        eta_radar_dB = results_fig5["Radar-only"]["eta_avg_dB"]

        plt.plot(
            Gamma_radar,
            eta_radar_dB,
            color="red",
            linestyle="-",
            marker="*",
            linewidth=2.8,
            markersize=11,
            label="Radar-only"
        )

    # ==================================================
    # Joint schemes
    # ==================================================

    for K in results_fig5.keys():

        if K == "Radar-only":

            continue

        Gamma_dB = results_fig5[K]["Gamma_dB"]

        eta_proposed_dB = results_fig5[K]["eta_proposed_avg_dB"]

        eta_similarity_dB = results_fig5[K]["eta_similarity_avg_dB"]

        eta_waveform22_dB = results_fig5[K]["eta_waveform22_avg_dB"]

        plt.plot(
            Gamma_dB,
            eta_proposed_dB,
            marker="o",
            linestyle="-",
            linewidth=2,
            label=rf"Proposed, $K={K}$"
        )

        plt.plot(
            Gamma_dB,
            eta_similarity_dB,
            marker="s",
            linestyle="--",
            linewidth=2,
            label=rf"Similarity P21, $K={K}$"
        )

        plt.plot(
            Gamma_dB,
            eta_waveform22_dB,
            marker="^",
            linestyle=":",
            linewidth=2,
            label=rf"Waveform P22, $K={K}$"
        )

    plt.xlabel(r"SINR constraint, $\Gamma_k$ (dB)")

    plt.ylabel("Average DPSL (dB)")

    plt.title(rf"Fig. 5: Radar-only, Proposed, P21 and P22, $N={N}$")

    plt.grid(True)

    plt.legend(fontsize=9)

    plt.tight_layout()

    plt.savefig("fig5_all_methods_with_radar_only.png", dpi=300)

    plt.show()


def myf_plot_fig6_convergence(results_fig6):


    plt.figure(figsize=(8, 5))

    marker_dict = {
        1: "*",
        2: "v",
        3: "s",
    }

    for K in results_fig6.keys():

        iteration = results_fig6[K]["iteration"]

        sinr_trace_dB = results_fig6[K]["sinr_trace_dB"]

        plt.plot(
            iteration,
            sinr_trace_dB,
            marker=marker_dict.get(K, "o"),
            linewidth=2,
            markersize=8,
            label=rf"Joint-$K={K}$"
        )

    plt.xlabel("Iteration number")

    plt.ylabel("SINR (dB)")

    plt.title(r"Fig. 6: SINR Convergence, $N=16$")

    plt.grid(True)

    plt.legend()

    plt.xlim([1, max(iteration)])

    plt.tight_layout()

    plt.savefig("fig6_sinr_convergence.png", dpi=300)

    plt.show()


def myf_plot_fig7_impairment(results_fig7):


    plt.figure(figsize=(8, 5))

    style_dict = {

        0: {
            "color": "red",
            "linestyle": "-",
            "label": r"$\kappa=0$"
        },

        0.03: {
            "color": "blue",
            "linestyle": "--",
            "label": r"$\kappa=0.03$"
        },

        0.06: {
            "color": "black",
            "linestyle": "-.",
            "label": r"$\kappa=0.06$"
        },

        0.09: {
            "color": "magenta",
            "linestyle": ":",
            "label": r"$\kappa=0.09$"
        },
    }

    for kappa in results_fig7.keys():

        theta = results_fig7[kappa]["theta"]

        pattern_dB = results_fig7[kappa]["pattern_dB"]

        style = style_dict.get(
            kappa,
            {
                "color": None,
                "linestyle": "-",
                "label": rf"$\kappa={kappa}$"
            }
        )

        plt.plot(
            theta,
            pattern_dB,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            label=style["label"]
        )

    plt.xlabel(r"$\theta$ (deg)")

    plt.ylabel("Beampattern (dBi)")

    plt.title(
        r"Fig. 7: Optimized Beampattern with Transmitter Impairment, "
        r"$N=16$, $\Gamma_k=20$ dB, $K=3$"
    )

    plt.grid(True)

    plt.legend()

    plt.xlim([-90, 90])

    plt.ylim([-35, 15])

    plt.tight_layout()

    plt.savefig("fig7_impairment_beampattern.png", dpi=300)

    plt.show()

def myf_plot_fig2_pareto_boundary(results_fig2):


    Gamma_dB = results_fig2["Gamma_dB"]

    eta_dB = results_fig2["eta_dB"]

    feasible = results_fig2["feasible"]

    valid_idx = np.logical_and(feasible, ~np.isnan(eta_dB))

    Gamma_valid = Gamma_dB[valid_idx]

    eta_valid = eta_dB[valid_idx]

    plt.figure(figsize=(8, 5))

    # ==================================================
    # Feasible achievable region
    # ==================================================

    if len(Gamma_valid) > 0:

        y_min = np.nanmin(eta_valid) - 1

        plt.fill_between(
            Gamma_valid,
            y_min,
            eta_valid,
            alpha=0.2,
            label="Achievable region"
        )

        # Pareto boundary
        plt.plot(
            Gamma_valid,
            eta_valid,
            marker="o",
            linewidth=2.5,
            label="Pareto boundary"
        )

        # Example ray-search direction
        idx_start = 0

        idx_end = len(Gamma_valid) - 1

        plt.annotate(
            "",
            xy=(Gamma_valid[idx_end], eta_valid[idx_end]),
            xytext=(Gamma_valid[idx_start], eta_valid[idx_start]),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=2,
                linestyle="--"
            )
        )

        plt.text(
            Gamma_valid[int(len(Gamma_valid) / 2)],
            eta_valid[int(len(eta_valid) / 2)] + 0.3,
            "Ray search direction",
            fontsize=10
        )

    plt.xlabel(r"SINR constraint, $\Gamma_k$ (dB)")

    plt.ylabel("Minimum DPSL (dB)")

    plt.title("Fig. 2: Pareto Boundary Illustration")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig("fig2_pareto_boundary.png", dpi=300)

    plt.show()