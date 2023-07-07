import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

import mplhep as hep

from functools import partial

from .analysis_util import *
from .plot_util import *

def plot_cutflow(out, hep_args=None, legend_title=None, filename=None):
    cutflow = out["cutflow"]
    fig, ax = plt.subplots()
    x = np.arange(len(cutflow))
    y = cutflow.values()
    xlabels = cutflow.keys()
    ax.step(x, y, "|-", where="post")
    ax.set_xticks(x, xlabels, rotation = "vertical")
    for i, v in enumerate(y):
        ax.text(x[i]+0.05, v + 100, str(v), fontsize=10)
        ax.axvline(x[i], linestyle="dashed", alpha=0.1)
    
    format_plot(fig, ax, legend_title=legend_title, legend_loc="upper right", 
                hep_args=hep_args, hep_magic=False,
                filename=filename)
    
# unify response, asymmetry, tp_response, tp_asymmetry, tp_metprojection
def plot_spectrum(out, hist_name,
                  dataset="QCD", jet_type=None,
                  x_jet_label=None, y_jet_label=None,
                  eta_range=None, phi_range=None,
                  normalize_pt=False, plot_what=["2d", "profile"],
                  labels=None,
                  fig=None, ax=None,
                  hline=None, vline=None,
                  xscale=None, yscale=None, xlim=None, ylim=None,
                  xlabel=None, ylabel=None, legend_title=None, 
                  hep_args=None, hep_magic=False,
                  save_plot_name=None,
                  filename=None):
    
    if isinstance(plot_what, str):
        plot_what = [plot_what]
    if isinstance(labels, str):
        labels = [labels]
    if labels is None:
        labels = [labels] * len(plot_what)
    assert len(labels) == len(plot_what), "number of plots requested must equal to number of labels"
    
    if fig == None or ax == None:
        fig, ax = plt.subplots()
        size = fig.get_size_inches()
        fig.set_size_inches((size[0]*1.5, size[1]))
    
    cmap = cm.get_cmap('rainbow').copy()
    cmap.set_under('w')
    
    h = preprocess_histogram(out[hist_name], dataset=dataset, 
                             jet_type=jet_type, eta_range=eta_range, phi_range=phi_range)

    h = h.project("jet_pt", hist_name) # this will integrate out any axis = None
    
    
    for i in range(len(plot_what)):
        # spectral plot
        if plot_what[i] == "2d":
            if normalize_pt:
                h_np = h.values() # convert to np
                h_np_norm = h_np / np.sum(h_np, axis=1, keepdims=True)
                h_norm = hist.Hist(*h.axes, storage=hist.storage.Double())
                h_norm.view()[:] = h_np_norm # convert back to hist
                h_norm.plot2d(ax=ax, cmap=cmap, alpha=1, norm=colors.LogNorm(), label=labels[i])
            else:
                h.plot2d(ax=ax, cmap=cmap, alpha=1, norm=colors.LogNorm())
        # compute profile and plot both mean and median
        elif plot_what[i] == "profile":
            # mean
            hp = h.profile(hist_name)
            ax.errorbar(hp.axes["jet_pt"].centers, hp.values(),
                        xerr=hp.axes["jet_pt"].widths/2,
                        yerr=np.sqrt(hp.variances()),
                        color="k", linestyle="", marker="o", markersize=4, label="mean")
            # median
            hp = profile_median(h, hist_name)
            #ax.step(hp.axes["jet_pt"].centers, hp.values(), where='mid', color="grey", linewidth=1.5)
            ax.errorbar(hp.axes["jet_pt"].centers, hp.values(),
                        xerr=hp.axes["jet_pt"].widths/2,
                        yerr=np.sqrt(hp.variances()),
                        color="grey", linestyle="", marker="o", markersize=4, label="median")
            
        # compute profile (mean) and plot arithmatic mean
        elif plot_what[i] == "mean":
            hp = h.profile(hist_name)
            #hp.plot(color="k", linewidth=1.5)
            #ax.step(hp.axes["jet_pt"].centers, hp.values(), where='mid', color="k", linewidth=1.5)
            ax.errorbar(hp.axes["jet_pt"].centers, hp.values(),
                        xerr=hp.axes["jet_pt"].widths/2,
                        yerr=np.sqrt(hp.variances()),
                        color="k", linestyle="", marker="o", markersize=4, label=labels[i])
        
        # compute median and plot median
        elif plot_what[i] == "median":
            hp = profile_median(h, hist_name)
            #ax.step(hp.axes["jet_pt"].centers, hp.values(), where='mid', color="grey", linewidth=1.5)
            ax.errorbar(hp.axes["jet_pt"].centers, hp.values(),
                        xerr=hp.axes["jet_pt"].widths/2,
                        yerr=np.sqrt(hp.variances()),
                        color="grey", linestyle="", marker="o", markersize=4, label=labels[i])
        
        # compute response = (1 + prof) / (1 - prof)
        elif plot_what[i] == "derive": # scale factor
            hp = h.profile(hist_name)
            centers = h.axes["jet_pt"].centers
            width = h.axes["jet_pt"].widths
            val = hp.values()
            resp = (1 + val) / (1 - val)
            resp_err = 2 * np.sqrt(hp.variances()) / (1 - val) ** 2 * resp
            ax.errorbar(centers, resp, yerr=resp_err, fmt="o", alpha=0.5, label=labels[i])
            
        else:
            raise ValueError("Unrecognized plot: {}".format(plot_what[i]))
    
    # plot horizontal lines, e.g. y=1 (response) or y=0 (asymmetry, metprojection)
    if hline is not None:
        if isinstance(hline, list):
            for line in hline:
                ax.axhline(y=line, linestyle="dotted", color="k", alpha=1)
        else:
            ax.axhline(y=hline, linestyle="dotted", color="k", alpha=1)
            
    # plot vertical lines
    if vline is not None:
        if isinstance(hline, list):
            for line in hline:
                ax.axvline(x=line, linestyle="dotted", color="k", alpha=1)
        else:
            ax.axvline(x=hline, linestyle="dotted", color="k", alpha=1)
    
    # x-axis and y-axis label (only default formatting for now...)
    if callable(xlabel):
        xlabel = xlabel(x_jet_label, y_jet_label)
    if callable(ylabel):
        ylabel = ylabel(x_jet_label, y_jet_label)
        
    legend_title_list = [dataset]
    if eta_range:
        legend_title_list.append(format_eta_range_text(eta_range[0], eta_range[1], 
                                                       jet_name=x_jet_label, omit_zero=True))
    if legend_title:
        legend_title = "\n".join([legend_title] + legend_title_list)
    else:
        legend_title = "\n".join(legend_title_list)
    
    if filename == -1 or filename == 0:
        save_dir = os.path.join(out["configurations"]["IO"]["output_dir"], "plot")
        mkdir_if_not_exist(save_dir)
        file_type = "pdf" if filename == -1 else "png"
        save_plot_name = save_plot_name if save_plot_name is not None else hist_name
        filename = get_default_filename(dataset, save_plot_name, jet_type, eta_range, 
                                        normalize_pt, xscale, yscale, file_type=file_type)
        filename = os.path.join(save_dir, filename)
    
    format_plot(fig, ax, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim,
                xlabel=xlabel, ylabel=ylabel, legend_title=legend_title, 
                hep_args=hep_args, hep_magic=hep_magic, filename=filename)
    
    return fig, ax

plot_response = partial(plot_spectrum, hist_name="response", 
                        xlabel=format_pt_text, ylabel=format_response_text,
                        hline=1.0)
plot_asymmetry = partial(plot_spectrum, hist_name="asymmetry", 
                         xlabel=format_ave_pt_text, ylabel=format_asymmetry_text,
                         hline=0.0)
plot_tp_response = partial(plot_spectrum, hist_name="tp_response", 
                           xlabel=format_pt_text, ylabel=format_response_text,
                           hline=1.0)
plot_tp_asymmetry = partial(plot_spectrum, hist_name="tp_asymmetry", 
                            xlabel=format_ave_pt_text, ylabel=format_asymmetry_text,
                            hline=0.0)
plot_tp_metprojection = partial(plot_spectrum, hist_name="tp_metprojection", 
                                xlabel=format_ave_pt_text, ylabel=format_metprojection_text,
                                hline=0.0)
plot_tp_db = partial(plot_spectrum, hist_name="tp_asymmetry", plot_what="derive",
                     xlabel=format_ave_pt_text, ylabel="Response $R_{rel}$",
                     hline=1.0)
plot_tp_mpf = partial(plot_spectrum, hist_name="tp_metprojection", plot_what="derive",
                      xlabel=format_ave_pt_text, ylabel="Response $R_{rel}$",
                      hline=0.0)