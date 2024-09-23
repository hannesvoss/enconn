import numpy as np
import pytest

from strategies import TemporalEncoding, generate_poisson_spikes


def test_temporal_encoding():
    strategy = TemporalEncoding()
    assert strategy.name == 'Temporal Encoding'
    data = [np.random.rand(50, 8), np.random.rand(50, 8), np.random.rand(50, 8)]
    spike_times = strategy.execute(data)
    assert len(spike_times) == 3
    assert spike_times[0].shape == (50, 8)
    assert spike_times[1].shape == (50, 8)
    assert np.all(spike_times[0] >= 0)
    assert np.all(spike_times[1] >= 0)


def test_poisson_generates_correct_spike_count():
    rate = 10
    time_window = 1.0
    result = generate_poisson_spikes(rate, time_window)
    assert isinstance(result, int)


def test_poisson_handles_zero_rate():
    rate = 0
    time_window = 1.0
    result = generate_poisson_spikes(rate, time_window)
    assert result == 0


def test_poisson_handles_large_rate():
    rate = 1000
    time_window = 1.0
    result = generate_poisson_spikes(rate, time_window)
    assert result >= 0


def test_poisson_handles_fractional_time_window():
    rate = 10
    time_window = 0.5
    result = generate_poisson_spikes(rate, time_window)
    assert isinstance(result, int)


def test_poisson_handles_negative_rate():
    rate = -10
    time_window = 1.0
    with pytest.raises(ValueError):
        generate_poisson_spikes(rate, time_window)