import copy
import json
import math
import os.path as osp


from experiments.table1.settings import TRAINING_MODES, DATASETS, MODELS, TRAINING_MODE_OBLIVIOUS_MODELS
from graphsage.settings import args


# region helper subroutines
def _post_process(results):

    for dataset in DATASETS:
        # Repeat performance across training mode for training mode oblivious models (as in the original paper)
        for model in set(TRAINING_MODE_OBLIVIOUS_MODELS) & set(MODELS):
            try:
                results[dataset]['unsupervised'][model] = results[dataset]['supervised'][model]
            except KeyError:
                pass
        # Add percentage f1 gain relative to raw features baseline
        for training_mode in TRAINING_MODES:
            for model in MODELS:
                try:
                    results[dataset][training_mode][model]['percentage_f1_gain'] = \
                        (results[dataset][training_mode][model]['test_f1'] -
                         results[dataset][training_mode]['raw_features']['test_f1']) / \
                        results[dataset][training_mode]['raw_features']['test_f1']
                except KeyError:
                    pass

    return results


def _datasets_section():
    section = ''

    for i, dataset in enumerate(DATASETS):
        section += f'\t\\multicolumn{{2}}{{c}}{{ {dataset.upper()} }}'

        if i < len(DATASETS) - 1:
            section += ' & '
        else:
            section += ' \\\\' + '\n'

    return section


def _models_section(mean_results, std_results, gains_acc):
    section = ''

    for i, model in enumerate(MODELS):
        section += f"\t{'-'.join(w.upper() for w in model.split('_'))} & "
        for j, dataset in enumerate(DATASETS):
            for k, training_mode in enumerate(TRAINING_MODES):
                try:
                    current_mean_f1 = mean_results[dataset][training_mode][model]['test_f1']
                    current_std_f1 = std_results[dataset][training_mode][model]['test_f1']
                    current_f1_str = f'{current_mean_f1:.3f}' + (f'\\tiny{{$\\pm {current_std_f1:.3f}$}}' if
                                                                 args.std else '')
                    sub_results_dict = {k: v for k, v in mean_results[dataset][training_mode].items() if k in MODELS}
                    all_f1s = [sub_results_dict[model]['test_f1'] for model in sub_results_dict]
                    if current_mean_f1 == max(all_f1s):
                        section += f'\\underline{{\\textbf{{{current_f1_str}}}}} & '
                        if 'percentage_f1_gain' in mean_results[dataset][training_mode][model]:
                            gains_acc[k + j * len(TRAINING_MODES)] = sub_results_dict[model]['percentage_f1_gain'] * 100
                        else:
                            gains_acc[k + j * len(TRAINING_MODES)] = 0
                    else:
                        section += f'{current_f1_str} & '
                except KeyError:
                    section += '--' + ' & '

        section = section[:-2] + '\\\\' + '\n'

    return section


# endregion

# def _initialize_dict(result):
#     return {
#         dataset: {
#             training_mode: {
#                 model: {k: 0 for k in result[dataset][training_mode][model].keys()}
#                 for model in result[dataset][training_mode]
#             } for training_mode in result[dataset].keys()
#         } for dataset in result.keys()
#     }


def compute_mean_std(results_list):
    N = len(results_list)

    mean_results = copy.deepcopy(results_list[0])
    std_results = copy.deepcopy(results_list[0])

    # Initialize mean and std results
    for dataset in mean_results.keys():
        for training_mode in mean_results[dataset].keys():
            for model in mean_results[dataset][training_mode].keys():
                for k in mean_results[dataset][training_mode][model].keys():
                    mean_results[dataset][training_mode][model][k] = 0
                    std_results[dataset][training_mode][model][k] = 0

    # Compute mean results
    for result in results_list:
        for dataset in mean_results.keys():
            for training_mode in mean_results[dataset].keys():
                for model in mean_results[dataset][training_mode].keys():
                    for k, v in mean_results[dataset][training_mode][model].items():
                        mean_results[dataset][training_mode][model][k] += result[dataset][training_mode][model][k]

    # Normalize mean results
    for dataset in mean_results.keys():
        for training_mode in mean_results[dataset].keys():
            for model in mean_results[dataset][training_mode].keys():
                for k in mean_results[dataset][training_mode][model].keys():
                    mean_results[dataset][training_mode][model][k] /= N

    # Compute std results
    for result in results_list:
        for dataset in std_results.keys():
            for training_mode in std_results[dataset].keys():
                for model in std_results[dataset][training_mode].keys():
                    for k in std_results[dataset][training_mode][model].keys():
                        std_results[dataset][training_mode][model][k] += (result[dataset][training_mode][model][k] - mean_results[dataset][training_mode][model][k]) ** 2

    # Normalize std results
    for dataset in std_results.keys():
        for training_mode in std_results[dataset].keys():
            for model in std_results[dataset][training_mode].keys():
                for k, v in std_results[dataset][training_mode][model].items():
                    std_results[dataset][training_mode][model][k] = math.sqrt(v / N)

    return mean_results, std_results


def generate_latex_table(results):
    results_list = [_post_process(result) for result in results]
    mean_results, std_results = compute_mean_std(results_list)

    num_cols = 1 + len(TRAINING_MODES) * len(DATASETS)
    gains_acc = [0] * (num_cols - 1)

    backslash = '\\'

    return f'''\\begin{{tabular}}{{{'l' + 'c' * (num_cols - 1)}}}
    \\hline
    \\multirow{{2}}{{*}}{{ Models (Ours) }} &
    {_datasets_section()}
    
    \\cline {{ 2 - {num_cols} }} &
        {(' Unsup. F1 & Sup. F1 & ' * ((num_cols - 1) // 2))[:-2]} \\\\
    
    \\hline
    {_models_section(mean_results, std_results, gains_acc)}
    
    \\hline
    \\% Gain over Raw Features & {' & '.join(f'{gain:.0f}{backslash}%' for gain in gains_acc)} \\\\
    \\hline
    \\end{{tabular}}'''


if __name__ == '__main__':
    results_dir = args.results_dir
    assert osp.exists(results_dir), f'{results_dir} does not exist'
    results = json.load(open(osp.join(results_dir, 'table1.json'), 'r'))
    if not isinstance(results, list):
        results = [results]
    print(generate_latex_table(results))
