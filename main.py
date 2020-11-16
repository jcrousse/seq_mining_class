from nn_model.nn_pattern_identification import get_multiple_patterns, run_experiment

if __name__ == '__main__':

        experiments = {
            "alpha_200_mp": {
                'seq_len': 500,
                'vocab_size': 1000,
                'multiple_patterns': get_multiple_patterns(20),
                'fp_rate': 0.05,
                'fn_rate': 0.05,
                'data_limit': 200
            },
        }
        for exp_name, exp_parameters in experiments.items():
            run_experiment(exp_name, **exp_parameters)
