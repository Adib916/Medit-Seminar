from __future__ import print_function
from __future__ import division
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import config
import microphone
import dsp

from midi import MidiConnector
from midi import NoteOn
from midi import Message

"""Needed variables for calculations"""
count = 0
i = 1
iteration = 0
used = 0

"""Connect port and set baudrate of Pi to send MIDI signals """
conn = MidiConnector('/dev/serial0', 38400)

"""The previous time that the frames_per_second() function was called"""
_time_prev = time.time() * 1000.0

"""The low-pass filter used to estimate frames-per-second"""
_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)


def frames_per_second():
    """Return the estimated frames per second"""
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


""" needed for scaling mel data after transforming fft into mel spectrum"""
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                              alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)


def microphone_update(audio_samples):
    """Called by microphone.py in while loop.
    Takes audio inpurt and:
    1. Calculates FFT
    2. transforms FFT into mel spectrum via scalar multiplying the mel filter bank with the FFT
    3. Creates rolling array y_roll_fft to capture the last config.N_ROLLING_FFT_HISTORY number of y[0], which is
    the 1st value of the scalar multiplication in step 2. This 1st value corresponds to the
    lowest band of the mel filter bank (lowest bandpass filter) multiplied with the fft, i.e. it
    captures the lowest frequency band as configured in config.py. This is useful for beat detection,
    as beats usually are in the lower frequency ranges.
    4. Calculates the mean and standard deviation of y_roll_fft and uses them for the first adaptive threshold
    calculation. Delayed threshold_calc by 2 * config.N_ROLLING_FFT_HISTORY iterations, because until then
     y_roll_fft consists only of 0s. After this it saves y_roll_fft in the new rolling array filteredY
    5. Uses exponential smoothing and saves the smoothed values into filteredY
    6. From now on mean and standard deviation will be calculated from filteredY.
    7. Use these values for adaptive threshold calculation in threshold_calc

    User has to tune config.THRESHOLD and config.INFLUENCE in order to get good results. Could be focus
    of future work."""

    global y_roll, prev_rms, prev_exp, count, i, threshold, influence, iteration
    global filteredY, avgFilter, stdFilter, signal_time
    iteration += 1

    # Normalize samples between 0 and 1
    y = audio_samples / 2.0**15

    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)

    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
    else:
        # Transform audio input into the frequency domain
        N = len(y_data)
        N_zeros = 2**int(np.ceil(np.log2(N))) - N
        # Pad with zeros until the next power of two
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:N // 2])

        # with open('logFFT.txt', 'a+') as file:
        #     file.write("%s\n" % (YS))

        # Construct a Mel filterbank from the FFT data
        mel = np.atleast_2d(YS).T * dsp.mel_y.T
        # Scale data to values more suitable for visualization
        # mel = np.sum(mel, axis=0)
        mel = np.sum(mel, axis=0)
        mel = mel**2.0
        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        y = mel

        if i == 1:
            for k in range(0, config.N_ROLLING_FFT_HISTORY - 1):
                y_roll_fft[k] = y[0]
                i = 0

        # # Rolling FFT window
        y_roll_fft[:-1] = y_roll_fft[1:]
        y_roll_fft[-1] = y[0]

        """Delayed threshold_calc by 2 * config.N_ROLLING_FFT_HISTORY iterations, because until then
         y_roll_fft consists only of 0s"""
        if iteration == 2 * config.N_ROLLING_FFT_HISTORY:
            filteredY = np.array(y_roll_fft)
            avgFilter = np.mean(y_roll_fft[0:config.N_ROLLING_FFT_HISTORY])
            stdFilter = np.std(y_roll_fft[0:config.N_ROLLING_FFT_HISTORY])
        elif iteration > 2 * config.N_ROLLING_FFT_HISTORY:
            threshold_calc(y[0])
        else:
            pass


def threshold_calc(CurrentValue):
    """Adaptive threshold calculation to adapt to different mel values or environments"""
    global filteredY, avgFilter, stdFilter, count, signal_time, used

    if abs(CurrentValue - avgFilter) > config.THRESHOLD * stdFilter:
        if CurrentValue > avgFilter:
            if count > 31:
                count = 0
            count += 1
            print("MIDI")
            print("\n")
            print("\n")
            print("\n")

            if used == 0:
                conn.write(Message(NoteOn(count, 69), 1))
                used = 1
        filteredTmp = config.INFLUENCE * CurrentValue + (1 - config.INFLUENCE) * filteredY[-1]

    else:
        used = 0
        filteredTmp = CurrentValue
        print("\n")

    filteredY[:-1] = filteredY[1:]
    filteredY[-1] = filteredTmp

    avgFilter = np.mean(filteredY[:(config.N_ROLLING_FFT_HISTORY)])
    stdFilter = np.std(filteredY[:(config.N_ROLLING_FFT_HISTORY)])


# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

# Array containing the rolling fft window
y_roll_fft = np.random.rand(config.N_ROLLING_FFT_HISTORY)


if __name__ == '__main__':
    # Start listening to live audio stream
    microphone.start_stream(microphone_update)
