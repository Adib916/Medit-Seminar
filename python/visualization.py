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

count = 0
i = 1
iteration = 0

conn = MidiConnector('/dev/serial0', 38400)

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""


def frames_per_second():
    """Return the estimated frames per second"""

    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


common_mode = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                            alpha_decay=0.99, alpha_rise=0.01)
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)

_prev_spectrum = np.tile(0.01, config.N_FFT_BINS)


mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                              alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()


def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update, _prev_spectrum, count, i, threshold, influence, iteration
    global filteredY, avgFilter, stdFilter, signal_time
    iteration += 1

    time.sleep(0.1)

    # Normalize samples between 0 and 1
    y = audio_samples / 2.0**15
    # with open('logAudio.txt', 'a+') as file:
    #     file.write("%s\n" % (y))
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

        if iteration == config.N_ROLLING_FFT_HISTORY:
            filteredY = np.array(y_roll_fft)
            avgFilter = np.mean(y_roll_fft[0:config.N_ROLLING_FFT_HISTORY])
            stdFilter = np.std(y_roll_fft[0:config.N_ROLLING_FFT_HISTORY])
        elif iteration > config.N_ROLLING_FFT_HISTORY:
            thresholding_algo(y[0])
        else:
            pass


'''https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/43512887#43512887'''


def thresholding_algo(CurrentValue):
    global filteredY, avgFilter, stdFilter, count, signal_time
    # print("Y0 is ", CurrentValue)
    # print("Y0 - - avgFilter ", CurrentValue-avgFilter)
    # print("config.THRESHOLD * stdFilter is ", config.THRESHOLD * stdFilter)
    if abs(CurrentValue - avgFilter) > config.THRESHOLD * stdFilter:
        if CurrentValue > avgFilter:
            if count > 31:
                count = 0
            count += 1
            print("___________________________________________________________________________count is ", count)
            print("\n")
            print("\n")
            print("\n")

            conn.write(Message(NoteOn(count, 69), 1))

        filteredTmp = config.INFLUENCE * CurrentValue + (1 - config.INFLUENCE) * filteredY[-1]

    else:
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
