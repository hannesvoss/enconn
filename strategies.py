import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def generate_poisson_spikes(rate, time_window=1.0):
    """Generates poisson spikes for a given firing rate and time window."""
    return np.random.poisson(rate * time_window, size=rate.shape)


class Strategy(object):
    """Strategy interface for encoding input data into spike times."""

    def __init__(self, name, show_plot=False, debug=False):
        self.name = name
        self.show_plot = show_plot
        self.debug = debug

    def execute(self, data):
        raise NotImplementedError

    def plot(self, spike_times):
        # plot the spike times as a heatmap
        plt.figure(figsize=(16, 10))
        heatmap_data = np.array([st.flatten() for st in spike_times])
        sns.heatmap(heatmap_data, cmap='viridis', cbar=True)
        plt.title(f'Spiking Activity Heatmap of \'{self.name}\' Input')
        plt.xlabel('Time steps')
        plt.ylabel('Neurons')
        plt.show()

    def debug_data(self, data):
        print(f'Debugging data for \'{self.name}\':')
        print(data)


class TemporalEncoding(Strategy):
    """Generate input in the form of temporal encoding using numpy poisson function."""

    def __init__(self, plot=False, debug=False):
        super().__init__('Temporal Encoding', plot, debug)

    def execute(self, data):
        spike_times = []
        for array in data:
            # Normalize the input data to the range [0, 1]
            normalized_data = (array - np.min(array)) / (np.max(array) - np.min(array))
            time_window = 1.0  # in seconds

            # Generate poisson spikes for the whole array at once
            spike_data = generate_poisson_spikes(normalized_data, time_window)

            spike_times.append(spike_data)
        return spike_times


class RankOrderEncoding(Strategy):
    """Generate input in the form of rank-order encoding using numpy argsort function."""

    def __init__(self, plot=False, debug=False):
        super().__init__('Rank Order Encoding', plot, debug)

    def execute(self, data):
        spike_times = []
        for array in data:
            # Rank the input values and assign spike times based on the rank
            ranks = np.argsort(np.argsort(array, axis=None)).reshape(array.shape)
            spike_times.append(ranks)
        return spike_times


class PopulationEncoding(Strategy):
    """Generate input in the form of population encoding."""

    def __init__(self, plot=False, debug=False):
        super().__init__('Population Encoding', plot, debug)

    def execute(self, data):
        spike_times = []
        for array in data:
            # Generate population encoding by distributing the input values across a population of neurons
            population_data = np.zeros_like(array)
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    # Scale to a range suitable for population encoding
                    population_data[i, j] = array[i, j] * 100

            spike_times.append(population_data)
        return spike_times
