import math
import multiprocessing
import os
import stat

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter, lfilter, savgol_filter
from scipy.signal import convolve
from scipy.ndimage import maximum_filter1d, minimum_filter1d, zoom, gaussian_filter1d

STANDARD_FS = 5
BPS_RANGE = np.array([8., 30.])

# emerald processing
def normalize_signals_percentile(signal: np.ndarray, p=1):
    floor = np.percentile(signal, p)
    cell = np.percentile(signal, 100 - p)
    signal = signal.clip(floor, cell)
    signal /= np.std(signal)
    return signal

def get_sleep_start_end(stages: np.ndarray):
    # stages should be an array with a frequency of 30 second per epoch
    stages_binary = np.clip(stages, 0, 1)
    for start, end in label_to_interval(stages_binary, 0):
        if end - start < 10 and start > 0 and end < len(stages):
            stages_binary[start:end] = 1
    for start, end in label_to_interval(stages_binary, 1):
        if end - start < 20:
            stages_binary[start:end] = 0
            break
    for start, end in label_to_interval(stages_binary, 0):
        if end - start < 240 and start > 0 and end < len(stages):
            stages_binary[start:end] = 1
    sleep_periods = sorted(label_to_interval(stages_binary, 1), key=lambda a: a[1] - a[0])
    if len(sleep_periods) > 0:
        sleep_start, sleep_end = sleep_periods[-1]
    else:
        sleep_start, sleep_end = 0, 1
    # plt.step(np.arange(len(stages)), stages, where="post")
    # plt.axvspan(sleep_start, sleep_end, color="orange", alpha=0.3)
    # plt.show()
    return sleep_start, sleep_end


def signal_snr(signal, fs=STANDARD_FS, with_std=True, crop_motion=False, show=False):
    from scipy.signal import spectrogram

    # signal is an array consisting of signal values for
    # one angle*bin*(real/imaginary) point over a sequence of frames
    # signal is indexed by frame number
    # e.g., shape might be (615,)

    # remove 10 seconds of data from the signal
    # following any high-energy entry in the signal

    # STANDARD_FS is 5 (5 frames/second)
    # return 0 if we have less than 30 seconds of data

    if len(signal) < 30 * fs:
        return 0

    # "Compute a spectrogram with consecutive Fourier transforms"
    # i.e., STFT, we'll apply a number of fourier transforms to the signal
    # sliding a window along
    # "spectrograms can be used as a way of visualizing the change of a nonstationary signal's frequency"

    # first parameter is input data, second is sampling frequency? (5 Hz)
    # third is "window" - a Tukey window is a tapered cosine window
    # flat at value 1 in the center, first and last 0.25/2 fraction of samples
    # at the beginning and end taking on cosine values
    # fourth argument is length of each segment, here 15 seconds, or 75 frames
    # fifth argument is number of points to overlap between segments, here 10 seconds, 50 frames
    # sixth argument is length of fft used, here 150 points (equivalent to 30 seconds)
    # seventh argument explains how to 'detrend' (?) each segment (here 'constant')
    # eighth argument specifies whether to return one-sided FFT results (here True)
    # i.e., return frequencies 0 through f, where f is Nyquist frequency
    # last argument explains that we want the absolute magnitude of the STFT

    # returns a triple: array of sample frequencies, array of segment times, spectrogram of signal

    # spectrogram is a two-dimensional ndarry, indexed by fft point, segment
    # shape of spec is, e.g., (76, 22)
    # 150-point one-sided fft returns 76 points
    # with 615 samples (123 seconds) in signal, and a new segment starting every 25 samples (5 seconds)
    # with segment length 75 (15 seconds) have segments starting at times (in seconds)
    # 7.5, 12.5, 17.5, ..., 107.5, 112.5
    # (22 segments)
    # segment frequencies: 0, 0.03333, 0.06667, 0.1, ... 2.5 Hz
    # increments are (2.5Hz/75) = 0.033333 Hz = 2 cycles/minute
    # (note that elsewhere at 5 samples per second, we computed ffts with 300 points rather than 150)

    _, _, spec = spectrogram(signal, fs, ('tukey', .25), 15 * fs, 10 * fs, 30 * fs, 'constant', True, mode='magnitude')

    # sum up the values in spec across the first dimension
    # (i.e., sum over all frequencies)
    # note that our DFT produced magnitudes, not complex numbers
    # energy_sum has shape, e.g., (22,)

    energy_sum = np.sum(spec, axis=0)

    # for each segment, find the index of the
    # frequency with the largest value
    # peak_index has shape, e.g., (22,)

    peak_index = np.argmax(spec, axis=0)

    # note that each index corresponds to 2 cycles per minute here,
    # whereas elsewhere it is 1 cycle per minute
    # zero out any peak indices where the frequency is below
    # the minimum breathing rate (e.g., 8)
    # or above the maximum breathing rate (e.g., 30)

    peak_index[np.where(peak_index * 2 < BPS_RANGE[0])] = 0

    peak_index[np.where(peak_index * 2 > BPS_RANGE[1])] = 0

    # initialize col_score to be an array of zeros with
    # length equal to number of segments

    col_score = np.zeros(spec.shape[1])

    # loop through all segments

    for i in range(spec.shape[1]):
        # for the i'th segment, sum up the values for frequencies between
        # peak_index[i]-1 and peak_index[i]+2
        # (what does python do if peak_index[i] == 0?)
        # we end up giving a range where the low index is -1
        # which is the last element in the array
        # while this will work, since the first index is larger than the second
        # it is UGLY, so just explicitly check for peak_index[i] > 0
        # note that we would never actually consider a breathing signal at 0 breaths/minute
        # as a legitimate peak

        if peak_index[i] > 0:
            col_score[i] = np.sum(spec[peak_index[i] - 1:peak_index[i] + 2, i])
            # also adding in value for double the frequency?
            col_score[i] += np.sum(spec[peak_index[i] * 2 - 1:peak_index[i] * 2 + 2, i])

    # presumably we add 1e-5 to avoid division by 0?
    # for each segment, divide "signal" by the total energy (the "noise")

    col_score /= energy_sum + 1e-5

    # over all segments, pick the median as snr the snr to return
    snr = np.median(col_score)

    if with_std:
        # by default we subtract a 1/100 fraction of the
        # standard deviation of peak frequency for each segment
        # from the snr?  Also note that peak_index represents
        # half the breathing rate in cycles/minute
        # but I guess the 0.01 constant is arbitrary anyway

        snr = snr - 0.01 * np.std(peak_index)

    if show:
        fig, ax = plt.subplots(4, 1, sharex='all')
        ax[0].plot(np.linspace(0, len(col_score), len(signal)), signal)
        spec_col = spec / np.max(spec, axis=0, keepdims=True)
        ax[1].imshow(spec_col, cmap='jet', aspect='auto', extent=[0, len(col_score), 150, 0])
        ax[1].invert_yaxis()
        ax[2].plot(col_score)
        ax[3].plot(peak_index)
        plt.suptitle(f'{np.median(col_score)}, {np.std(peak_index)}')
        plt.show()
    return snr, col_score


def detect_static_signal(signal, fs=10):
    from scipy.ndimage.filters import minimum_filter1d
    signal = signal.copy()
    a = np.concatenate([np.zeros((1,)), signal], axis=0)
    sig_diff = np.diff(a, axis=0)
    threshold = 0.02
    window = 30
    static_part = sig_diff < threshold
    static_part = minimum_filter1d(static_part, int(window * fs))
    indices = np.where(static_part == 1)[0]
    non_static_indices = np.where(static_part == 0)[0]

    return static_part, 1 - static_part, non_static_indices


def signal_crop(signal, clip_limit=6):
    signal = np.clip(signal, -clip_limit, clip_limit)
    return signal


def norm_sig(input_sig):
    return (input_sig - np.mean(input_sig)) / np.std(input_sig)


def detect_motion_iterative(signal, fs=10, level=3):
    signal = signal.copy()
    motion = np.ones(len(signal), dtype=int)
    right_most_ratio = 1
    if level == 0 or len(signal) < 30 * fs:
        std = signal_std(signal)
        signal = signal / std
        right_most_ratio = 1 / std
        motion *= 0
    else:
        signal_crop, indices = signal_crop_motion(signal, window=10, threshold=10, fs=fs)
        if level == 3 and len(signal_crop) == len(signal):
            signal_crop, indices = signal_crop_motion(signal, window=10, threshold=6, fs=fs)
        motion[indices] = 0
        stable_periods = label_to_interval(motion, 0)
        for i, (p0, p1) in enumerate(stable_periods):
            signal_norm, right_r, motion_seg = detect_motion_iterative(signal[p0: p1], level=level - 1)
            signal[p0: p1] = signal_norm
            motion[p0: p1] = motion_seg
            if i != len(stable_periods) - 1:
                signal[p1:stable_periods[i + 1][0]] *= right_r
            else:
                right_most_ratio = right_r
    signal = np.clip(signal, -8, 8)
    return signal, right_most_ratio, motion


def my_detect_motion_iterative(signal, fs=10, level=3):
    signal = signal.copy()
    motion = np.ones(len(signal), dtype=int)
    right_most_ratio = 1
    if level == 0 or len(signal) < 30 * fs:
        std = signal_std(signal)
        signal = signal / std
        right_most_ratio = 1 / std
        motion *= 0
    else:
        signal_crop, indices = signal_crop_motion(signal, window=10, threshold=10, fs=fs)
        if level == 3 and len(signal_crop) == len(signal):
            signal_crop, indices = signal_crop_motion(signal, window=10, threshold=6, fs=fs)
        motion[indices] = 0
        stable_periods = label_to_interval(motion, 0)
        for i, (p0, p1) in enumerate(stable_periods):
            signal_norm, right_r, motion_seg = my_detect_motion_iterative(signal[p0: p1], level=level - 1)
            signal[p0: p1] = signal_norm
            motion[p0: p1] = motion_seg
            if i != len(stable_periods) - 1:
                signal[p1:stable_periods[i + 1][0]] *= right_r
            else:
                right_most_ratio = right_r

    # signal = np.clip(signal, -6, 6)
    for l, r in label_to_interval(np.abs(signal) >= 6, 1):
        l = max(l - 30 * fs, 0)
        r = min(r + 30 * fs, len(signal))
        signal[l:r] = 0

    return signal, right_most_ratio, motion


def signal_std(signal):
    if len(signal) < 10:
        return 1
    else:
        cut = int(len(signal) * 0.1)
        std = np.std(np.sort(signal)[cut:-cut])
    std = 1 if std == 0 else std
    return std


def signal_normalize(signal):
    signal -= np.mean(signal)
    return signal / signal_std(signal)


def signal_crop_motion(signal, window=10, fs=10, threshold=5):
    from scipy.ndimage.filters import minimum_filter1d
    signal_norm = signal_normalize(signal)
    threshold = max(np.max(np.abs(signal_norm)) * 0.5, threshold)
    normal_part = np.abs(signal_norm) < threshold
    normal_part = minimum_filter1d(normal_part, int(window * fs))
    indices = np.where(normal_part == 1)[0]
    signal_crop = signal_norm[indices]
    return signal_crop, indices


def label_to_interval(label: np.array, val=0):
    hit = (label == val).astype(int)
    a = np.concatenate([np.zeros((1,)), hit.flatten(), np.zeros((1,))], axis=0)
    a = np.diff(a, axis=0)
    left = np.where(a == 1)[0]
    right = np.where(a == -1)[0]
    return np.array([*zip(left, right)], dtype=np.int32)


def zoom_complex(input, ratio):
    real_part = input.real
    imag_part = input.imag

    zoom_real = zoom(real_part, ratio)
    zoom_imag = zoom(imag_part, ratio)
    out = np.zeros(zoom_real.shape, dtype=np.complex64)
    out.real = zoom_real
    out.imag = zoom_imag
    return out


def compute_local_std_mean0(length, input_data):
    assert length % 2 == 0
    # speeding up algorithm
    ave_kernel = np.ones((length,), dtype='float32') / length
    local_mean = convolve(input_data, ave_kernel, mode='same')
    residual = input_data - local_mean
    residual_square = residual ** 2
    local_std = convolve(residual_square, ave_kernel, mode='same') ** 0.5 + 1e-30
    return np.divide(residual, local_std)


def makedir_with_oursmode(filename):
    """make dirs and let it"""
    os.makedirs(filename, exist_ok=True)
    chmod_ours(filename)


def chmod_ours(filename):
    """change file mode to user and group rwx"""
    os.chmod(filename, stat.S_IRWXG | stat.S_IRWXU)


def iterate_funct(funct, filelist, processed_path, out_suffix):
    for each in filelist:
        funct(each, processed_path, out_suffix)


def parse_raw_data(parse_icml_data, process_num, from_file_list, target_file):
    file_list = sorted(os.listdir(from_file_list))
    L = len(file_list)
    step = (L // process_num) + 1
    args = [(file_list[i * step:min((i + 1) * step, L)], target_file,) for i in range(process_num)]
    multiprocess(parse_icml_data, args, process_num)


def generate_data(f, process_num, file_path, from_file_list, suffix):
    file_list = sorted(os.listdir(file_path + from_file_list))
    L = len(file_list)
    step = (L // process_num) + 1
    args = [(f, file_list[i * step:min((i + 1) * step, L)], file_path, suffix,) for i in range(process_num)]
    multiprocess(iterate_funct, args, process_num)


def save_parsed(filename, savepath, **kwargs):
    for name, data in kwargs.items():
        np.savez_compressed(os.path.join(savepath, name, filename), data=data[0], fs=data[1])


def multiprocess(func, args, process_num):
    for i in range(process_num):
        p1 = multiprocessing.Process(target=func, args=args[i])
        p1.start()


def standardize_stage(stages, epoch_length=30, fs=10):
    """Standardize stage labels: mapping and reshaping"""
    stage_labels = stage_mapping(stages)
    stage_labels = np.repeat(stage_labels, epoch_length * fs, axis=0)
    return stage_labels


def standardize_apnea(apnea_events, signal_length, fs=10):
    """Standardize event labels: mapping and transformation"""
    apnea_labels = np.zeros(signal_length, dtype=np.int32)
    for event_name, start, duration in apnea_events:
        label = apnea_mapping(event_name)
        apnea_labels[int(start * fs):int((start + duration) * fs)] = label
    return apnea_labels


def standardize_arousal(apnea_events, length, fs=256):
    """Standardize event labels: mapping and transformation"""
    arousal_labels = np.zeros(length, dtype=np.int32)
    for event_name, start, duration in apnea_events:
        arousal_labels[int(start * fs):int((start + duration) * fs)] = 1
    return arousal_labels


def standardize_desaturation(apnea_events, length, fs=1):
    """Standardize event labels: mapping and transformation"""
    desat_labels = np.zeros(length, dtype=np.int32)
    for event_name, start, duration, desat in apnea_events:
        desat_labels[int(start * fs):int((start + duration) * fs)] = int(desat)
    return desat_labels


def trend(x, window_length=51, polyorder=2):
    return savgol_filter(x, window_length, polyorder)


sample_rate = 10
win_len = int(sample_rate * 15) * 2 + 1


def detrend(x, window_length=win_len, polyorder=2):
    return x - savgol_filter(x, window_length, polyorder)


def stage_mapping(stages):
    stages[stages == 4] = 3
    stages[stages == 5] = 4
    stages[stages > 4] = 0  # handle error case (e.g., 6)
    return stages.astype(np.int32)


def apnea_mapping(event_name):
    if event_name == 'Hypopnea':
        return 1
    elif event_name == 'Obstructive Apnea':
        return 2
    elif event_name == 'Central Apnea':
        return 3
    elif event_name == 'Mixed Apnea':
        return 4
    else:
        raise Exception('Error: unknown type of apnea/hypopnea (%s)!' % event_name)


def butter_bandpass_filter(data, cutoff1, cutoff2, fs, order=5):
    nyq = 0.5 * fs
    cutoff1 = cutoff1 / nyq
    cutoff2 = cutoff2 / nyq
    b, a = butter(order, [cutoff1, cutoff2], btype='bandpass', analog=False)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    cutoff = cutoff / nyq
    b, a = butter(order, [cutoff], btype='lowpass', analog=False)
    y = lfilter(b, a, data)
    return y


def interpol_signal(s, old_f, new_f, type='slinear'):
    if old_f == new_f:
        return s
    x = np.arange(s.shape[0])
    if type in ['slinear', 'quadratic', 'nearest']:
        f = interpolate.interp1d(x, s, kind=type)
    elif type == 'spline':
        f = interpolate.UnivariateSpline(x, s)
    else:
        raise RuntimeWarning("Unknown type of interpolation.")
    tempf = f(np.arange(0, s.shape[0] - 1, old_f / new_f))
    if old_f > new_f:
        return tempf
    elif 2 * old_f > new_f:
        return np.append(tempf, s[-1])
    else:
        return np.append(tempf, [s[-1], s[-1]])


def interpol_signal_2d(s, old_f, new_f, type='quadratic'):
    if old_f == new_f:
        return s
    x = np.arange(s.shape[1])
    for row in range(s.shape[0]):
        s_row = s[row, :]
        if type in ['slinear', 'quadratic', 'nearest']:
            f = interpolate.interp1d(x, s_row, kind=type)
        elif type == 'spline':
            f = interpolate.UnivariateSpline(x, s_row)
        else:
            raise RuntimeWarning("Unknown type of interpolation.")
        tempf = f(np.arange(0, s_row.shape[0] - 1, old_f / new_f))

        if 2 * old_f > new_f:
            tempf = np.append(tempf, s_row[-1])
        else:
            tempf = np.append(tempf, [s_row[-1], s_row[-1]])
        if row == 0:
            out = np.zeros([s.shape[0], tempf.shape[0]], dtype='float32')
        out[row, :] = tempf
    return out


def one_runs(a):
    # pad each end with an extra 0.
    iszero = np.concatenate(([0], a, [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def AHI(apnea_label, stage_label, signal_size):
    """get apnea_num and sleep hours"""
    # get apnea_num
    if signal_size > 0:
        batch_num = math.ceil(apnea_label.shape[0] / signal_size) * 2
        step = math.floor((apnea_label.shape[0] - signal_size) / (batch_num - 1))
    else:
        batch_num = 1
        step = apnea_label.shape[0]
        signal_size = apnea_label.shape[0]

    temp = np.logical_and(apnea_label > 0, stage_label > 0)
    temp = one_runs(temp)
    nums = np.zeros(batch_num)
    hours = np.zeros(batch_num)
    for i in range(nums.shape[0]):
        for j in range(temp.shape[0]):
            if temp[j, 1] - temp[j, 0] < 10 * 10:  # 10 seconds
                continue
            # totally included
            if temp[j, 0] >= i * step and temp[j, 1] <= i * step + signal_size:
                nums[i] += 1
            # first part of apnea event included
            elif i * step <= temp[j, 0] <= i * step + signal_size:
                nums[i] += (i * step + signal_size - temp[j, 0]) / (temp[j, 1] - temp[j, 0])
            # last part of apnea event included
            elif i * step <= temp[j, 1] <= i * step + signal_size:
                nums[i] += (temp[j, 1] - i * step) / (temp[j, 1] - temp[j, 0])

    # get sleep_hours
    for i in range(nums.shape[0]):
        hours[i] = (stage_label[i * step: (i * step + signal_size)] > 0).sum() / 36000

    return nums.astype('float32'), hours.astype('float32')


def pptid2poi(visit, pptid):
    """Generate poi string from visit and poi"""
    return str(visit) + str(pptid)


def poi2pptid(poi):
    """Generate visit and poi strings from pptid"""
    poi_str = str(poi)
    return poi_str[:1], poi_str[1:]


def compute_local_std_1d(length, input_data):
    """
    :param length: window size to compute std
    :param input_data: input sequence
    :return: normalized input sequence, the windowed std of the sequence
    """
    assert length % 2 == 0
    # speeding up algorithm
    ave_kernel = np.ones((length,), dtype='float32') / length
    local_mean = convolve(input_data, ave_kernel, mode='same')
    residual = input_data - local_mean
    residual_square = residual ** 2
    local_std = convolve(residual_square, ave_kernel, mode='same') ** 0.5 + 1e-30
    return np.divide(input_data, local_std), local_std


def _whole_spec_eeg(eeg_data, nfft_window_size, step, win_num=30, freq_threshold=32):
    """
    :param win_num: seg eeg signal into patches for multi cpus, in minutes
    :param eeg_data:
    :param freq_threshold:
    :return:
    """
    signal_cut_length = win_num * nfft_window_size * EEG_RAW_FS
    total_length = eeg_data.shape[0]
    pieces = total_length // signal_cut_length
    pad_length = EEG_RAW_FS * ((nfft_window_size - step) // 2)
    assert (nfft_window_size - step) % 2 == 0

    for i in range(pieces):
        front = i * signal_cut_length - pad_length
        rear = (i + 1) * signal_cut_length + pad_length

        if i == pieces - 1:
            eeg_slice = eeg_data[i * signal_cut_length:]
            eeg_cut = np.concatenate((eeg_data[front: i * signal_cut_length],
                                      eeg_slice,
                                      np.zeros((pad_length,), dtype=np.float64)), axis=0)
        else:
            eeg_slice = eeg_data[i * signal_cut_length: (i + 1) * signal_cut_length]
            if i == 0:
                assert rear <= total_length
                eeg_cut = np.concatenate((np.zeros((pad_length,), dtype=np.float64),
                                          eeg_slice,
                                          eeg_data[(i + 1) * signal_cut_length:rear]), axis=0)
                # pad 0 in front
            else:
                if rear <= total_length:
                    eeg_cut = np.concatenate((eeg_data[front: i * signal_cut_length],
                                              eeg_slice,
                                              eeg_data[(i + 1) * signal_cut_length:rear]), axis=0)
                    # pad 0 in front and rear
                else:
                    padding_needed = rear - total_length
                    eeg_slice = eeg_data[i * signal_cut_length:]

                    eeg_cut = np.concatenate((eeg_data[front: i * signal_cut_length],
                                              eeg_slice,
                                              np.zeros((padding_needed,), dtype='float32')), axis=0)

        result = get_spec_eeg(eeg_cut, nfft_window_size, step, is_pad=False, freq_threshold=freq_threshold)

        power = np.zeros([EEG_SPEC_HEIGHT, result.shape[1]], dtype='float32')
        freq_step = (EEG_SPEC_MAX_FREQ * nfft_window_size) / EEG_SPEC_HEIGHT

        for row in range(EEG_SPEC_HEIGHT):
            power[row] = np.sum(result[int(row * freq_step):int((row + 1) * freq_step - 1), :] ** 2, axis=0) ** 0.5

        if i == 0:
            spec = power
        else:
            spec = np.concatenate((spec, power), axis=1)
    return spec


def get_spec_eeg(eeg_data, nfft_window_size, step, is_pad=True, freq_threshold=32):
    if is_pad:
        eeg_data = np.concatenate((np.zeros(EEG_RAW_FS * ((nfft_window_size - step) // 2), dtype=np.float64),
                                   eeg_data, np.zeros(EEG_RAW_FS * ((nfft_window_size - step) // 2),
                                                      dtype=np.float64)), axis=0)
    nfft = nfft_window_size * EEG_RAW_FS
    step = step * EEG_RAW_FS
    freq_index_max = 1 + nfft_window_size * freq_threshold
    from scipy.signal import spectrogram
    _, _, s = spectrogram(x=eeg_data, window='hann', nperseg=nfft,
                          noverlap=nfft - step, nfft=nfft,
                          detrend='linear', mode='magnitude')
    s[1:] += s[1:][::-1]
    result = s[1: freq_index_max, :]
    return result
