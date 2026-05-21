import matplotlib.pyplot as plt

'''
fPlot.py
'''


def myf_plot_fig3(sys_param, fig3_result, output_path="figures/paper_fig3.png"):

    ### Functions ###

    fig, axes = plt.subplots(2, 1, figsize=(5.8, 8.0))
    color_cand = ["r", "b", "k", "r"]
    line_cand = ["-", "--", "-.", ":"]
    label_cand = ["Radar-only", r"Joint-$K$ = 1", r"Joint-$K$ = 2", r"Joint-$K$ = 3"]

    for ind1 in range(0, 2):
        result_temp = fig3_result[ind1]
        for ind2 in range(0, 4):
            axes[ind1].plot(result_temp["theta_grid_plot"], result_temp["P_dBi"][ind2],
                            color_cand[ind2] + line_cand[ind2], linewidth=1.2, label=label_cand[ind2])
        axes[ind1].grid()
        axes[ind1].legend(loc="upper right", fontsize=8)
        axes[ind1].set_xlabel(r'$\theta$(deg)')
        axes[ind1].set_ylabel(r'Beampattern (dBi)')
        axes[ind1].set_xlim(-90, 90)
        if (result_temp["N"] == 12):
            axes[ind1].set_ylim(-70, 20)
            axes[ind1].text(0.5, -0.25, r'(a) The optimized beampatterns, $N=12$, $\Gamma_k=20$ dB, $K=0,1,2,3$.',
                            ha="center", va="top", transform=axes[ind1].transAxes, fontsize=8)
        elif (result_temp["N"] == 16):
            axes[ind1].set_ylim(-60, 20)
            axes[ind1].text(0.5, -0.25, r'(b) The optimized beampatterns, $N=16$, $\Gamma_k=20$ dB, $K=0,1,2,3$.',
                            ha="center", va="top", transform=axes[ind1].transAxes, fontsize=8)

    fig.tight_layout(h_pad=3.0)
    fig.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig)
