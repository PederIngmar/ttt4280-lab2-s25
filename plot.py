import numpy as np
import matplotlib.pyplot as plt
import os
import raspi_import as rpi
import config

def plot(outdir, filename, data):
    """
    Plot data from `raspi_import` in a 3x1 grid for the last 3 channels.
    """
    data = data[:, 2:]  # Start from the 3rd channel (index 2)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    
    total_samples = data.shape[0]
    time_axis = np.linspace(0, 1, total_samples)

    # Define colors for each plot
    colors = [config.colors[f'mic{i+1}'] for i in range(3)]

    for i, ax in enumerate(axs):
        chandata = data[:, i]
        ax.plot(time_axis[250:], chandata[250:], color=colors[i])
        ax.set_ylabel(f'adc {i+1}')
        ax.set_xlim(0, 1)  # Set xlim to zoom in

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Samplet lydsignal gjennom 3 ADCs')

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    plt.savefig(os.path.join(outdir, f'{filename}.png'))
    #plt.show()

if __name__ == "__main__":
    # Example usage
    angles = [0, 36, 75, 90, 110, 150, 180, 270]
    for angle in angles:
        outdir = f"plots/{angle}/"
        input_dir = f"data/{angle}"
        files, datas = rpi.import_all_files(input_dir)
        for i, data in enumerate(datas):
            plot(outdir, files[i], data)