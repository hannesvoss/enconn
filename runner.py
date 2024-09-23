import os
from datetime import datetime

import conn2res
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from conn2res import readout
from conn2res.connectivity import Conn
from conn2res.readout import Readout
from conn2res.reservoir import SpikingNeuralNetwork
from matplotlib import pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame


class Runner:
    def __init__(self, name, seed, task, connectomes, encoding_strategies, evaluation_metrics, project_dir,
                 data_dir=None, output_dir=None):
        self.name = name
        self.seed = seed
        self.task = task
        self.set_seed(seed)
        self.alphas = np.linspace(0, 2, 9)[1:]
        self.connectomes = connectomes
        self.encoding_strategies = encoding_strategies
        self.evaluation_metrics = evaluation_metrics
        self.project_dir = project_dir
        self.data_dir = os.path.join(self.project_dir, 'data') if data_dir is None else data_dir
        self.output_dir = os.path.join(
            self.project_dir,
            'results',
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ) if output_dir is None else output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        print(f'conn2res version installed: {conn2res.__version__}')
        print(f'Initialize with connectomes: {self.connectomes}')

    def run(self, n_trials=4050, input_gain=0.0001):
        print(f'{self.name} is running')
        x, y = self.task.fetch_data(n_trials=n_trials, input_gain=input_gain)

        # Save the input and output data
        np.save(os.path.join(self.output_dir, 'input.npy'), x)
        np.save(os.path.join(self.output_dir, 'output.npy'), y)

        all_metrics = pd.DataFrame()

        for strategy in self.encoding_strategies:
            print(f'Running experiment for {strategy.name}')
            spike_times = strategy.execute(x)
            if strategy.show_plot:
                strategy.plot(spike_times)
            if strategy.debug:
                strategy.debug_data(spike_times)

            df_metrics = pd.DataFrame()
            for connectome in self.connectomes:
                print(f'Running experiment for {connectome}')
                df_res = self._run_experiment(connectome, spike_times, y)
                df_res['Strategy'] = strategy.name
                df_res['Connectome'] = connectome
                df_metrics = pd.concat([df_metrics, df_res], ignore_index=True)

            all_metrics = pd.concat([all_metrics, df_metrics], ignore_index=True)

        all_metrics.to_csv(os.path.join(self.output_dir, f'all_metrics.csv'), index=False)

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _load_data(self, connectome):
        print(f'Loading connectome data for {connectome}')
        w = np.loadtxt(os.path.join(self.data_dir, connectome, 'conn.csv'), delimiter=',', dtype=float)
        labels = pd.read_csv(os.path.join(self.data_dir, connectome, 'labels.csv'))['Sensory'].values
        print(f'Connectome shape:  {w.shape}')
        print(f'Labels shape:  {labels.shape}')
        return w, labels

    def _run_workflow(self, w, x, y, input_nodes, output_nodes, rewire=False) -> DataFrame:
        conn = Conn(w=w)
        if rewire:
            conn.randomize(swaps=10)

        conn.scale_and_normalize()

        w_in = np.zeros((8, conn.n_nodes))
        w_in[:, input_nodes] = np.eye(1)

        snn = SpikingNeuralNetwork(w=conn.w)  # activation_function='tanh'
        readout_module = Readout(estimator=readout.select_model(y))

        x_train, x_test, y_train, y_test = readout.train_test_split(x, y)

        df_metrics = pd.DataFrame()
        for alpha in self.alphas:
            snn.w = alpha * conn.w

            rs_train = snn.simulate(
                ext_input=x_train, w_in=w_in,
                output_nodes=output_nodes
            )

            rs_test = snn.simulate(
                ext_input=x_test, w_in=w_in,
                output_nodes=output_nodes
            )

            df_res = readout_module.run_task(
                X=(rs_train, rs_test),
                y=(y_train, y_test),
                sample_weight=None,
                metric=self.evaluation_metrics,
                readout_modules=None,
                readout_nodes=None,
            )

            # print evaluation results for each evaluation_metric
            for metric in self.evaluation_metrics:
                print(f'{metric}: {df_res[metric][0]}')

            df_res['alpha'] = np.round(alpha, 3)
            df_metrics = pd.concat([df_metrics, df_res], ignore_index=True)

        return df_metrics

    def _run_experiment(self, connectome, x, y):
        # load the connectome data
        w, labels = self._load_data(connectome)

        # run workflow for empirical network
        return self._run_workflow(
            w.copy(), x, y,
            input_nodes=np.where(labels == 1)[0],
            output_nodes=np.where(labels == 0)[0],
        )

    def plot_metrics(self, metrics_file='all_metrics.csv'):
        # Load the metrics from the CSV file
        metrics_path = os.path.join(self.output_dir, metrics_file)
        metrics_df = pd.read_csv(metrics_path)

        # Convert the Connectome column to a categorical type
        metrics_df['Connectome'] = metrics_df['Connectome'].astype('category')

        # Set up the matplotlib figure
        sns.set(style='whitegrid')
        num_metrics = len(self.evaluation_metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(14, 8 * num_metrics))

        # Plot each metric
        for i, metric in enumerate(self.evaluation_metrics):
            ax = sns.scatterplot(
                x='Connectome',
                y=metric,
                hue='Strategy',
                data=metrics_df,
                style='Strategy',
                s=100,
                ax=axes[i],
            )
            ax.set_title(f'Comparison of {metric} across Connectomes and Strategies')
            ax.set_xlabel('Connectome')
            ax.set_ylabel(metric)
            ax.set_xticklabels(metrics_df['Connectome'].cat.categories, rotation=45, horizontalalignment='right')
            ax.set_xticks(range(len(metrics_df['Connectome'].cat.categories)))
            ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

        plt.show()
