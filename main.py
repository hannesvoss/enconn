import os

from runner import Runner
from strategies import TemporalEncoding, RankOrderEncoding, PopulationEncoding
from task import OneStepInference

if __name__ == '__main__':
    experiment = Runner(
        name='CrossSpeciesInferenceExperiment',
        seed=42,
        task=OneStepInference(
            name='OneStepInferenceTask',
            goal_presentation_steps=20,
            delay_steps=10,
            choices_presentation_steps=20,
        ),
        encoding_strategies=[
            TemporalEncoding(),
            RankOrderEncoding(),
            PopulationEncoding(),
        ],
        connectomes=[
            'drosophila',
            'macaque_modha',
            'mouse',
            'rat'
        ],
        evaluation_metrics=[
            # 'r2_score',
            'mean_squared_error',
            # 'root_mean_squared_error',
            # 'mean_absolute_error',
            # 'corrcoef',
            'accuracy_score',
            # 'balanced_accuracy_score',
            # 'f1_score',
            # 'precision_score',
            # 'recall_score',
        ],
        project_dir=os.getcwd(),
    )

    experiment.run(
        n_trials=50,
    )

    experiment.plot_metrics()
