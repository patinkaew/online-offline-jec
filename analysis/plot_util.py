import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

import mplhep as hep

# strike text
def strike(text):
    result = ""
    for c in text:
        result = result + c + "\u0336"
    return result

# return range text as str
def format_range_text(lower, upper, var, absolute=False, omit_zero=True):
    if lower > upper:
        lower, upper = upper, lower
    if absolute:
        var = "|{}|".format(var)
    if omit_zero and lower == 0:
        return "{} < {}".format(var, upper)
    else:
        return "{} < {} < {}".format(lower, var, upper)

# quick pre-configured for eta range 
def format_eta_range_text(lower, upper, jet_name="jet", absolute=True, omit_zero=False):
    return format_range_text(lower, upper, "$\eta^{%s}$"%(jet_name), absolute=absolute, omit_zero=omit_zero)

# quick pre-configured for pt range 
def format_pt_range_text(lower, upper, jet_name="jet", omit_zero=True):
    return format_range_text(lower, upper, "$p_T^{%s}$"%(jet_name), omit_zero=omit_zero)

# quick pre-configured for phi range 
def format_phi_range_text(lower, upper, jet_name="jet", omit_zero=False):
    return format_range_text(lower, upper, "$\phi^{%s}$"%(jet_name), omit_zero=omit_zero)

def format_pt_text(x_jet_label, y_jet_label="", full=False): # for consistency
    return r"$p_T^{%s}$"%x_jet_label if not full else r"$%s \ p_T$"%x_jet_label

def format_ave_pt_text(x_jet_label, y_jet_label, style=0, full=False):
    fmt_dict = {0:r"$(p_T^{%s} + p_T^{%s})/2$", 
                1:r"$0.5*(p_T^{%s} + p_T^{%s})$",
                2:r"$(%s \ p_T + %s \ p_T)/2$", 
                3:r"$%0.5*(%s \ p_T + %s \ p_T)$"}
    prefix = "" if not full else "Average "
    return prefix + fmt_dict[style]%(x_jet_label, y_jet_label)

def format_response_text(x_jet_label, y_jet_label, style=0, full=False):
    fmt_dict = {0:r"$R = \frac{p_T^{%s}}{p_T^{%s}}$", 
                1:r"$R = p_T^{%s} / p_T^{%s}$",
                2:r"$R = \frac{%s\ p_T}{%s\ p_T}$", 
                3:r"$R = %s\ p_T / %s\ p_T$"}
    prefix = "" if not full else "Response "
    return prefix + fmt_dict[style]%(y_jet_label, x_jet_label)

def format_asymmetry_text(x_jet_label, y_jet_label, style=0, full=False):
    fmt_dict = {0:r"$A = \frac{p_T^{%s} - p_T^{%s}}{p_T^{%s} + p_T^{%s}}$", 
                1:r"$A = (p_T^{%s} - p_T^{%s})/(p_T^{%s} + p_T^{%s})$",
                2:r"$A = \frac{%s\ p_T - %s\ p_T}{%s\ p_T + %s\ p_T}$", 
                3:r"$A = (%s\ p_T - %s\ p_T)/(%s\ p_T + %s\ p_T)$"}
    prefix = "" if not full else "Asymmetry "
    return prefix + fmt_dict[style]%(y_jet_label, x_jet_label, y_jet_label, x_jet_label)

def format_metprojection_text(x_jet_label, y_jet_label, style=0, full=False):
    fmt_dict = {0:r"$B = \frac{MET^{%s} \cdot \hat{p}_T^{%s}}{p_T^{%s} + p_T^{%s}}$", 
                1:r"$B$ = (MET^{%s} \cdot \hat{p}_T^{%s})/(p_T^{%s} + p_T^{%s})$",
                2:r"$B = \frac{%s\ MET \cdot %s\ \hat{p}_T}{%s\ p_T + %s\ p_T}$", 
                3:r"$B = (%s\ MET \cdot %s\ \hat{p}_T)/(%s\ p_T + %s\ $p_T$)$"}
    prefix = "" if not full else "MET Projection "
    # met label is extracted by removing ", tag" or ", probe", if any
    # and "AK4" or "AK8", if any
    return prefix + fmt_dict[style]%(y_jet_label.split(",")[0].replace("AK4", "").replace("AK8", ""), 
                                     x_jet_label, y_jet_label, x_jet_label)

def format_eta_range_save_text(eta_range, interpret=True):
    eta_bin_coarse = [0, 1.3, 2.5, 3, 5]
    eta_bin_range_name = ["BB", "EC1", "EC2", "HF"]
    if eta_range is None:
        return "inclusive"
    if len(eta_range) == 2:
        eta_range = eta_range + (False, )
    if interpret and len(eta_range) == 3:
        if eta_range[0] == -1.3 and eta_range[1] == 1.3:
            return "BB"
        for i in range(len(eta_bin_coarse)-1):
            if eta_range[0] == eta_bin_coarse[i] and eta_range[1] == eta_bin_coarse[i+1]:
                text = eta_bin_range_name[i]
                if eta_range[2] == False:
                    text = text + "p"
                return text
            elif eta_range[0] == -eta_bin_coarse[i+1] and eta_range[1] == -eta_bin_coarse[i]:
                text = eta_bin_range_name[i]
                if eta_range[2] == False:
                    text = text + "n"
                return text
    # now default
    text = "{}to{}".format(str(eta_range[0]).replace(".", "p").replace("-", "~"), 
                           str(eta_range[1]).replace(".", "p").replace("-", "~"))
    if eta_range[2]:
        text = "[{}]".format(text)
    return text

def get_default_filename(dataset, hist_name, jet_type, eta_range, 
                         normalize_pt=False, xscale=None, yscale=None, file_type="pdf"):
    xscale = xscale if xscale is not None else "linear"
    yscale = yscale if yscale is not None else "linear"
    fname = "-".join([dataset, hist_name, jet_type, "eta={}".format(format_eta_range_save_text(eta_range)),
                  "normalize_pt=%s"%normalize_pt, "xscale=%s"%xscale])
    fname = fname + ".{}".format(file_type.lower())
    return fname

# save figure to file
def save_figure(fig, filename=None, dpi=100):
    if filename:
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)

# wrapper to call appropriate matplotlib functions
def format_plt_plot(ax,
                    xscale=None, yscale=None,
                    xlim=None, ylim=None,
                    xlabel=None, ylabel=None,
                    legend_title=None, legend_loc=0, legend_args=None):
    
    # axis scaling
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    
    # setting axis limit
    # left=None, right=None, *, emit=True, auto=False, xmin=None, xmax=None
    if xlim:
        if isinstance(xlim, set) or isinstance(xlim, tuple):
            ax.set_xlim(*xlim)
        if isinstance(xlim, dict):
            ax.set_xlim(**xlim)
    if ylim:
        if isinstance(ylim, set) or isinstance(ylim, tuple):
            ax.set_ylim(*ylim)
        if isinstance(ylim, dict):
            ax.set_ylim(**xlim)
    
    # setting x-axis and y-axis label
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel: 
        ax.set_ylabel(ylabel)
        
    # formatting legend
    if legend_title:
        ax.legend(title=legend_title, loc=legend_loc)
    if legend_args: # other arguments, see matplotlib document for reference
        if legend_title and "title" not in legend_args:
            legend_args["title"] = legend_title
        if legend_loc and "loc" not in legend_args:
            legend_args["loc"] = legend_loc
        ax.legend(**legend_args)

# wrapper to call appropriate mplhep functions
def format_hep_plot(ax,
                    hep_args=None, with_cms_name=True,
                    hep_magic=False):
    if hep_args:
        if with_cms_name:
            hep.cms.label(**hep_args)
        else: # experimental, x, y, and fontsize may need to be adjusted manually
            if hep_args["data"]:
                ax.set_title("private", style="italic", ha="left", x=0, y=1.005, fontsize=28)
            else:
                ax.set_title("Simulation private", style="italic", ha="left", x=0, y=1.005, fontsize=28)
                
    if hep_magic:
        try:
            hep.mpl_magic(ax=ax)
        except:
            warnings.warn("mplhep magic fails! trying yscale_legend")
            try:
                hep.plot.yscale_legend(ax=ax)
            except:
                warnings.warn("mplhep yscale also fails!")

# combine both wrappers and save figure
# this is generally called at the end
def format_plot(fig, ax,
                xscale=None, yscale=None,
                xlim=None, ylim=None,
                xlabel=None, ylabel=None,
                legend_title=None, legend_loc=0, legend_args=None, 
                hep_args=None, with_cms_name=True,
                hep_magic=False,
                filename=None, dpi=100):
    
    format_plt_plot(ax, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,
                    legend_title=legend_title, legend_loc=legend_loc, legend_args=legend_args)
    
    if hep_args or hep_magic:
        format_hep_plot(ax, hep_args=hep_args, with_cms_name=with_cms_name, hep_magic=hep_magic)
        
    save_figure(fig, filename, dpi)