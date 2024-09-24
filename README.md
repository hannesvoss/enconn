# Encoding Strategies in Connectome-Informed Spiking Neural Networks (enconn)

This project is designed to run experiments on spiking neural networks using various encoding strategies and evaluation metrics. The experiments are conducted on different connectomes, and the results are saved and visualized.

## Project Structure

- `main.py`: The main entry point for running the experiments. It sets up the experiment parameters and runs the experiment.
- `strategies.py`: Contains different encoding strategies for generating input data for the spiking neural network.
- `tests.py`: Contains unit tests for the encoding strategies.
- `runner.py`: Contains the `Runner` class, which handles the execution of the experiments, including data loading, running the workflow, and plotting the results.

## How to Run the Code

1. **Install Dependencies**: Ensure you have Python installed. Install the required packages using pip:
    ```sh
    pip3 install -r requirements.txt
    ```

2. **Run the Experiment**: Execute the `main.py` file to start the experiment:
    ```sh
    python main.py
    ```

3. **View Results**: The results will be saved in the `results` directory within the project directory. You can view the metrics and plots generated during the experiment.

## File Descriptions

- **`main.py`**: 
  - Sets up the experiment parameters such as the number of trials, input gain, and evaluation metrics.
  - Runs the experiment and plots the metrics.

- **`strategies.py`**: 
  - Contains different encoding strategies:
    - `TemporalEncoding`: Generates input using temporal encoding.
    - `RankOrderEncoding`: Generates input using rank-order encoding.
    - `PopulationEncoding`: Generates input using population encoding.
  - Utility functions for generating Poisson input and spikes.

- **`tests.py`**: 
  - Contains unit tests for the encoding strategies to ensure they work as expected.

- **`runner.py`**: 
  - Contains the `Runner` class, which:
    - Initializes the experiment with the given parameters.
    - Loads data and connectomes.
    - Runs the workflow for each encoding strategy and connectome.
    - Saves and plots the results.

## Example Usage

To run the experiment with specific parameters, modify the `main.py` file as needed. For example, you can change the number of trials:
```python
experiment.run(
    n_trials=10,  # Number of trials
)