import pyedflib as edf
import numpy as np
import matplotlib.pyplot as plt


def read_file(filename: str):
    """
    Reads a .edf file into an EdfReader object and an array containing signals.

    :param filename: name of .edf file to read from. File must be in path ./Data/filename
    :return: EdfReader object and np.array containing signals.
    """
    # reading edf file
    f = edf.EdfReader(f'Data/baseline_{filename}.edf')
    # preparing to construct signals array
    n_signals = f.signals_in_file
    signals = np.zeros((n_signals, f.getNSamples()[0]))

    # constructing signals array
    for i in np.arange(n_signals):
        signals[i, :] = f.readSignal(i)

    return f, signals


def visualize(data: np.ndarray, signal: int, signal_name: str):
    """
    Visualizing **data** on the specified **signal** s

    :param data: Numpy array containing signal data - should be in the form such that rows are signal no. and columns
    are measurements.
    :param signal: Which signal to visualize. Fx 0
    """
    # TODO: some exception handling if signals are greater than number of signals
    # TODO: maybe pass EdfReader object and call a construct method. This way, we also know the name of the signal we're
    #   visualizing

    # Since we're only working with 1 signal, axis will be x = time, y = amplitude
    plot_data = data[signal]
    plt.plot(plot_data)
    # defining x and y axis, i.e. shape and label.
    plt.xlim(0, len(plot_data))
    plt.ylim((plot_data.min() * 1.05), (plot_data.max() * 1.05))  # 105% to get little extra in both ends - looks better
    plt.xlabel = 'Time'
    plt.ylabel = 'Amplitude'
    # Naming the plot according to signal_name
    plt.title(signal_name)
    plt.legend()
    plt.show()


def channel_index(edf_obj: edf.EdfReader, channel_name: str):
    """
    Returns the channel index of **channel_name** in **edf_obj** .

    :param edf_obj: EdfReader object to find channel in.
    :param channel_name: Name of the channel searched for.
    """
    labels = edf_obj.getSignalLabels()
    idx = 0
    for label in labels:
        # removing .'s
        label = label.replace('.', '')
        if channel_name.lower() == label.lower():
            return idx
        idx += 1

    raise Exception(f'{channel_name} not found as a signal label!')


def exercise_1(signal_data: np.ndarray, signal_number, signal_name):
    visualize(signal_data, signal_number, signal_name)


def exercise_2(edf_obj: edf.EdfReader, signal_data: np.ndarray):
    c3_idx = channel_index(edf_obj, 'c3')
    cz_idx = channel_index(edf_obj, 'cz')
    c4_idx = channel_index(edf_obj, 'c4')
    mrcp_data = signal_data[[c3_idx, cz_idx, c4_idx]]

    return mrcp_data


def exercise_3(signal_data: np.ndarray):
    # averaging each row, i.e. channel
    row_means = np.mean(signal_data, axis=1)
    # reshaping from (n,) -> (n,1) to make following calculations easier
    row_means = row_means.reshape((len(row_means), 1))

    return signal_data - row_means


if __name__ == '__main__':
    #closed eyes
    closed_edf, closed_signals = read_file('closedeyes')
    # open eyes
    open_edf, open_signals = read_file('openeyes')

    # Ex. 1)
    # exercise_1(closed_signals, 0, closed_edf.getLabel(0))

    # Ex 2)
    # As I understand it, simply get recordings from channels: C3, Cz and C4
    # mrcp_data = exercise_2(closed_edf, closed_signals)

    # Ex 3)
    x = exercise_3(closed_signals)

