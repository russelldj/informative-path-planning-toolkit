from ubelt import ensuredir
from pathlib import Path
import matplotlib.pyplot as plt
from ipp_toolkit.config import FIG_SIZE, PAUSE_DURATION


def show_or_save_plt(savepath=None, pause_duration=None, fig_size=FIG_SIZE):
    """
    Can either save to a file, pause, or showG
    """
    if savepath is not None:
        plt.gcf().set_size_inches(*fig_size)
        savepath = Path(savepath)
        plt.tight_layout()
        ensuredir(savepath.parent)
        plt.savefig(savepath)
        plt.close()
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
