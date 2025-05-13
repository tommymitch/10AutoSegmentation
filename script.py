import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import ximu3csv

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

devices = ximu3csv.read(path, ximu3csv.DataMessageType.INERTIAL)

seconds = devices[0].inertial.timestamp / 1e6
seconds -= seconds[0]

sample_rate = 1 / np.median(np.diff(seconds))

signal = np.linalg.norm(devices[0].inertial.accelerometer.xyz, axis=1)


def my_fft(
    signal: np.ndarray,
    fft_size: int,  # in number of samples (power of 2)
    hop_size: int,  # in number of samples
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [frequencies, time steps]
    window = np.hanning(fft_size)

    number_of_windows = int((len(signal) - fft_size) / (fft_size - hop_size)) + 1

    stft = np.empty((fft_size, number_of_windows), dtype=complex)

    for index in range(number_of_windows):
        start = index * (fft_size - hop_size)  # why is ths not index * hop_size
        end = start + fft_size

        stft[:, index] = np.fft.fft(signal[start:end] * window)

    times = (np.arange(number_of_windows) * (fft_size - hop_size)) / sample_rate

    frequencies = np.fft.fftfreq(fft_size, 1.0 / sample_rate)[: fft_size // 2]  # only positive frequencies
    dbs = 20 * np.log10(np.abs(stft[: fft_size // 2, :]))

    return (times, frequencies, dbs)


times, frequencies, dbs = my_fft(signal, 64, 16)

if False:
    # Plot the magnitude of the STFT
    plt.pcolormesh(times, frequencies, dbs, shading="auto")
    plt.title("STFT Magnitude (Multiple FFTs)")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")

    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()

THRESHOLD = 4
IMPACT_HOLDOFF = int(sample_rate / 5)  # 200 ms

if True:
    figure, axes = plt.subplots(nrows=2, sharex=True)

    axes[0].pcolormesh(times, frequencies, dbs, shading="auto")

    axes[1].plot(seconds, signal)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("g")
    axes[1].plot([seconds[0], seconds[-1]], [THRESHOLD, THRESHOLD])

# Detect impacts
impact_detection = signal > THRESHOLD
impact_detection = np.maximum.reduce([np.roll(impact_detection, i) for i in range(IMPACT_HOLDOFF + 1)], axis=0)  # extend true values forward in time by IMPACT_HOLDOFF period

# Create list of reps, one per impact
impact_starts = np.where((impact_detection & ~np.roll(impact_detection, 1)))[0]  # index of each false to true transition
impact_ends = impact_starts + IMPACT_HOLDOFF

for impact_start, impact_end in zip(impact_starts, impact_ends):
    axes[1].fill_between([seconds[impact_start],seconds[impact_end]],np.min(signal), np.max(signal),color="tab:green",alpha=0.2)
    print(impact_end - impact_start)

plt.show()
