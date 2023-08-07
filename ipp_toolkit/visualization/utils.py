from ubelt import ensuredir
from pathlib import Path
import matplotlib.pyplot as plt
from ipp_toolkit.config import SMALL_FIG_SIZE, MED_FIG_SIZE, PAUSE_DURATION

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1


# taken from
# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def show_or_save_plt(
    savepath=None,
    pause_duration=None,
    fig_size=SMALL_FIG_SIZE,
    _run=None,
    close=True,
):
    """
    Can either save to a file, pause, or showG
    """
    if savepath is not None:
        plt.gcf().set_size_inches(*fig_size)
        savepath = Path(savepath)
        plt.tight_layout()
        ensuredir(savepath.parent)
        plt.savefig(savepath)
        if close:
            plt.close()
        if _run is not None:
            _run.add_artifact(savepath)
        return

    if pause_duration is not None:
        plt.pause(pause_duration)
        plt.close()
        return

    plt.show()


def remove_ticks():
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
