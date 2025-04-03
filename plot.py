import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import detrend

def plot(outdir, filename, data):
    """
    Plot data from `raspi_import` in a 5x1 grid.
    """
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8))
    
    total_samples = data.shape[0]
    time_axis = np.linspace(0, 1, total_samples)

    for i, ax in enumerate(axs):
        chandata = data[:, i]
        ax.plot(time_axis[250:], chandata[250:])
        ax.set_ylabel(f'adc {i+1}')
        ax.set_xlim(0, 1)  # Set xlim to zoom in

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Sampled signal through 5 ADCs')

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    plt.savefig(os.path.join(outdir, f'{filename}.png'))
    #plt.show()
