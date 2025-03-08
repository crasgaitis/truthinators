import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from flask import Flask, render_template, Response
import openai

# Import functions from the utils module
from utils import update_buffer, get_last_data, compute_band_powers, julia, create_custom_colormap
from pylsl import StreamInlet, resolve_byprop

if __name__ == '__main__':
    BUFFER_LENGTH = 5
    EPOCH_LENGTH = 1
    OVERLAP_LENGTH = 0
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
    INDEX_CHANNEL = [0]

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    else:
        print('Found it!')
        print(streams)
        
    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info
    info = inlet.info()
    fs = int(info.nominal_srate())

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    print('Press Ctrl-C in the console to break the while loop.')

    fig, ax = plt.subplots()

    while True:
        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))

        # Only keep the channel we're interested in
        ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

        # Update EEG buffer with the new data
        eeg_buffer, filter_state = update_buffer(
            eeg_buffer, ch_data, notch=True,
            filter_state=filter_state)

        # Get newest samples from the buffer
        data_epoch = get_last_data(eeg_buffer,
                                            EPOCH_LENGTH * fs)

        # Compute band powers
        band_powers = compute_band_powers(data_epoch, fs)
        band_buffer, _ = update_buffer(band_buffer,
                                                np.asarray([band_powers]))
        print(band_buffer)