import numpy as np 
import os
import glob
from scipy.signal import detrend
from scipy.interpolate import interp1d
import raspi_import as rpi
import matplotlib.pyplot as plt


def interpolate_signals(signal1, signal2, sample_period, oversample_factor=10, kind='linear'):
    """
    Interpolate two signals to a finer time resolution.
    """
    # Original time axis
    t_orig = np.arange(len(signal1)) * sample_period

    # Build interpolators
    f1 = interp1d(t_orig, signal1, kind=kind)
    f2 = interp1d(t_orig, signal2, kind=kind)

    # New, finer time axis
    n_fine = len(signal1) * oversample_factor
    t_fine = np.linspace(t_orig[0], t_orig[-1], n_fine)

    # Evaluate upsampled signals
    u1 = f1(t_fine)
    u2 = f2(t_fine)

    return u1, u2, t_fine


def delays(sample_period, data, oversample_factor=10, kind='linear'):
    """
    Compute time delays between channels with sub‐sample precision
    by upsampling the windowed signals.

    oversample_factor: how many times finer to make the time axis
    kind: interpolation type ('linear','cubic', etc.)
    """
    if data is None:
        return None
    data = data[:, 2:]
    peak_idx = np.argmax(data[:, 0])
    start = max(0, peak_idx - 100)
    end = min(len(data[:, 0]), peak_idx + 2000)

    D = {}
    channel_pairs = [(0, 1), (1, 2), (0, 2)]

    for cpair in channel_pairs:
        d1 = detrend(data[:, cpair[0]])
        d2 = detrend(data[:, cpair[1]])

        w1 = d1[start:end]
        w2 = d2[start:end]

        u1, u2, t_fine = interpolate_signals(w1, w2, sample_period, oversample_factor, kind)

        cc = np.correlate(u1, u2, mode='full')
        lags = np.arange(-len(u1) + 1, len(u1)) * (t_fine[1] - t_fine[0])

        max_lag = lags[np.argmax(cc)]
        D[cpair] = max_lag

    return D


def plot_crosscorrelation(data, sample_period=3.2e-05, oversample_factor=10, kind='linear'):
    if data is None:
        return None
    data = data[:, 2:]
    peak_idx = np.argmax(data[:, 0])
    start = max(0, peak_idx - 100)
    end = min(len(data[:, 0]), peak_idx + 2000)
    channel_pairs = [(0, 1), (1, 2), (0, 2)]

    plt.figure(figsize=(10, 6))
    for cpair in channel_pairs:
        d1 = detrend(data[:, cpair[0]])
        d2 = detrend(data[:, cpair[1]])

        w1 = d1[start:end]
        w2 = d2[start:end]

        u1, u2, t_fine = interpolate_signals(w1, w2, sample_period, oversample_factor, kind)

        crosscorrelation = np.correlate(u1, u2, mode='full')
        lags = np.arange(-len(u1) + 1, len(u1)) * (t_fine[1] - t_fine[0])

        max_idx = np.argmax(crosscorrelation)
        max_lag = lags[max_idx]
        max_value = crosscorrelation[max_idx]

        plt.plot(lags, crosscorrelation, label=f"Kanal {cpair[0]+1} og {cpair[1]+1}")

        plt.axvline(x=max_lag, color='black', linestyle='--', alpha=0.7)
        #plt.text(max_lag, 0, f"{max_lag:.2f}", color='r', fontsize=10, ha='center')

    plt.title("Krysskorrelasjon mellom kanaler 1, 2, 3")
    plt.xlabel("Lag (sekunder)")
    plt.ylabel("Krysskorrelasjon")
    plt.grid()
    plt.xlim(-0.0008, 0.0006)  # Adjust x-axis limits to zoom in
    plt.legend()
    plt.savefig("plots/crosscorrelation_combined_zoomed.png")
    plt.show()

def estimate_angle(delays):
    t12, t13, t23 = delays[(1, 2)], delays[(0, 2)], delays[(0, 1)]

    x = (t13 - t12 + 2*t23)
    y = (t13 + t12)

    theta = np.arctan2(np.sqrt(3)* y, x)

    cali_factor = 123
    theta_deg = np.degrees(theta) + cali_factor

    return theta_deg


def plot_angle_results(real_angles, estimated_angles):
    """
    Plot estimated angles vs real angles, along with errors and basic stats.
    """
    real_angles = np.array(real_angles)
    estimated_angles = np.array(estimated_angles)
    errors = abs(estimated_angles - real_angles)

    # Plot estimated vs real
    plt.figure(figsize=(10, 6))
    plt.plot(real_angles, estimated_angles, 'o-', label='Estimert vinkel')
    plt.plot(real_angles, real_angles, '--', label='Ideell linje (y=x)')
    plt.xlabel("Reell vinkel (grader)")
    plt.ylabel("Estimert vinkel (grader)")
    plt.title("Estimert vs. reell vinkel")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/summary/angle_estimates.png")
    plt.show()

    # Plot error
    plt.figure(figsize=(10, 6))
    plt.plot(real_angles, errors, 'ro', label='Avvik')  # No lines between points
    plt.axhline(0, color='gray', linestyle='--')

    # Fit a regression polynomial to the errors
    coeffs = np.polyfit(real_angles, errors, deg=2)  # Fit a 2nd-degree polynomial
    poly = np.poly1d(coeffs)
    regression_x = np.linspace(min(real_angles), max(real_angles), 500)
    regression_y = poly(regression_x)
    plt.plot(regression_x, regression_y, 'b-', label='Regresjonspolynom')

    plt.xlabel("Reell vinkel (grader)")
    plt.ylabel("Avvik (grader)")
    plt.title("Avvik mellom estimert og reell vinkel med regresjonspolynom")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/summary/angle_errors.png")
    plt.show()

    # Print some stats
    print("Gjennomsnittlig avvik: {:.2f} grader".format(np.mean(errors)))
    print("Standardavvik i estimert vinkel: {:.2f}".format(np.std(estimated_angles)))


if __name__ == "__main__":
    angles = [0, 36, 75, 110, 150, 180, 270]
    real_angles = []
    """
    estimated_angles = []

    for angle in angles:
        input_dir = f"data/{angle}"
        files, datas = rpi.import_all_files(input_dir)
        for i, data in enumerate(datas):
            d = delays(3.2e-05, data)
            est_angle = estimate_angle(d)
            real_angles.append(angle)
            estimated_angles.append(est_angle)
            print("Real angle:", angle, "Estimated angle:", est_angle, "Difference:", est_angle - angle, "File", files[i])
    """
    # Etter alt er prosessert:
    #plot_angle_results(real_angles, estimated_angles)
    sample_period, data = rpi.raspi_import("data/0", "d-14.33.43")
    data = rpi.detrend_all(data)
    plot_crosscorrelation(data, 3.2e-05)




def compute_statistics():
    angles = [0, 36, 72, 108, 144, 180]
    estimated_angles = []
    print("Statistics for all angles:")
    for angle in angles:
        print(f"------- angle:{angle}-------")
        bin_files = glob.glob(f"output/{angle}/*.bin")
        for i in range(0, len(bin_files)):
            current_bin_file = bin_files[i]
            file_name = os.path.splitext(os.path.basename(current_bin_file))[0]
            sample_period, data = rpi.raspi_import(f"output/{angle}", file_name)
            d = delays(sample_period, data)
            estimated_angles.append(estimate_angle(d))
        print(f"Mean: {np.mean(estimated_angles):.2f}")
        print(f"Standard deviation: {np.std(estimated_angles):.2f}")
        print(f"Variance: {np.var(estimated_angles):.2f}")
        estimated_angles.clear()

def all_delays(data_list, sample_period=3.2e-05,):
    all_delays = []
    for data in data_list:
        d = delays(sample_period, data)
        all_delays.append(d)
    return all_delays

def average_delays(all_delays):
    avg_delays = {}
    for cpair in all_delays[0].keys():
        avg_delays[cpair] = np.mean([d[cpair] for d in all_delays])
    return avg_delays


def std_delays(all_delays):
    # Calculate variance of delays
    var_delays = {}
    for cpair in all_delays[0].keys():
        var_delays[cpair] = np.var([d[cpair] for d in all_delays])

    # Calculate standard deviation of delays
    std_delays = {}
    for cpair in all_delays[0].keys():
        std_delays[cpair] = np.std([d[cpair] for d in all_delays])

    return std_delays
