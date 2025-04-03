import numpy as np 
import os
import glob
from scipy.signal import detrend
import raspi_import as rpi
import matplotlib.pyplot as plt

def delays(sample_period, data):
    if data is None:
        return None
    data = data[:, 2:]
    peak_idx = np.argmax(data[:, 0])
    start = max(0,peak_idx-100)
    end = min(len(data[:, 0]),peak_idx+2000)
    D = {}
    channel_pairs = [(0, 1), (1, 2), (0, 2)]
    for cpair in channel_pairs:
        d1 = detrend(data[:, cpair[0]])
        d2 = detrend(data[:, cpair[1]])
        d1_window = d1[start:end]
        d2_window = d2[start:end]

        crosscorrelation = np.correlate(d1_window, d2_window, mode='full')
        lags = np.arange(-len(d1_window)+1, len(d1_window))

        max_lag = lags[np.argmax(crosscorrelation)]
        delta_t = max_lag*sample_period

        D[cpair] = delta_t

    return D

def plot_crosscorrelation(data, sample_period=3.2e-05):
    if data is None:
        return None
    data = data[:, 2:]
    peak_idx = np.argmax(data[:, 0])
    start = max(0, peak_idx - 100)
    end = min(len(data[:, 0]), peak_idx + 2000)
    channel_pairs = [(0, 1), (1, 2), (0, 2)]

    for cpair in channel_pairs:
        d1 = detrend(data[:, cpair[0]])
        d2 = detrend(data[:, cpair[1]])
        d1_window = d1[start:end]
        d2_window = d2[start:end]

        crosscorrelation = np.correlate(d1_window, d2_window, mode='full')
        lags = np.arange(-len(d1_window) + 1, len(d1_window))

        plt.figure()
        plt.plot(lags * sample_period, crosscorrelation)
        plt.title(f"Cross-correlation between channels {cpair[0]} and {cpair[1]}")
        plt.xlabel("Lag (seconds)")
        plt.ylabel("Cross-correlation")
        plt.grid()
        plt.xlim(-0.0006, 0.0006)  # Adjust x-axis limits to zoom in
        plt.show()

def estimate_angle(delays):
    #t12, t13, t23 = delays[(0, 1)], delays[(0, 2)], delays[(1, 2)]
    #t12, t13, t23 = delays[(1, 2)], delays[(0, 1)], delays[(0, 2)]
    #t12, t13, t23 = delays[(0, 1)], delays[(1, 2)], delays[(0, 2)]
    #t12, t13, t23 = delays[(0, 2)], delays[(1, 2)], delays[(0, 1)]
    #t12, t13, t23 = delays[(0, 2)], delays[(0, 1)], delays[(1, 2)]
    t12, t13, t23 = delays[(1, 2)], delays[(0, 2)], delays[(0, 1)]

    x = (t13 - t12 + 2*t23)
    y = (t13 + t12)

    theta = np.arctan2(np.sqrt(3)* y, x)
    
    cali_factor = 123
    theta_deg = np.degrees(theta) + cali_factor

    return theta_deg




if __name__ == "__main__":
    angles = [0, 36, 75, 110, 150, 180, 270] # 0, 36, 75, 110, 
    for angle in angles:
        input_dir = f"data/{angle}"
        files, datas = rpi.import_all_files(input_dir)
        for i, data in enumerate(datas):
            print("File:", files[i])
            d = delays(3.2e-05, data)
            #plot_crosscorrelation(data)
            #d = delays(3.2e-05, data)
            est_angle = estimate_angle(d)
            print("Real angle:", angle, "Estimated angle:", est_angle, "Difference:", est_angle - angle)








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
