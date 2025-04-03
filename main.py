from datetime import datetime
import raspi_import as rpi
import os
import dataget as dg
import plot as plot

def sample_and_download(angle):
    """
    Sample ADC data, download it, and plot the results.
    """
    inputdir = f"data/{angle}"
    outdir = f"plots/{angle}"
    os.makedirs(inputdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    filename = datetime.now().strftime('d-%H.%M.%S')
    print(filename)
    dg.remote_sample_adc(filename)
    dg.download_file(f"lab1", inputdir, filename, filename)
    return filename

if __name__ == "__main__":
    angle = 180
    local_dir = f"data/{angle}"
    filename = sample_and_download(angle)
    filename, data = rpi.import_latest_file(local_dir)
    plot_dir = f"plots/{angle}"
    plot.plot(plot_dir, filename, data)