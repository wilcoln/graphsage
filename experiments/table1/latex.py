import json
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


def _models_section(results, gains_acc):
    section = ''

    for i, model in enumerate(MODELS):
        section += f"\t{'-'.join(w.upper() for w in model.split('_'))} & "
        for j, dataset in enumerate(DATASETS):
            for k, training_mode in enumerate(TRAINING_MODES):
                try:
                    current_f1 = results[dataset][training_mode][model]['test_f1']
                    sub_results_dict = {k: v for k, v in results[dataset][training_mode].items() if k in MODELS}
                    all_f1s = [sub_results_dict[model]['test_f1'] for model in sub_results_dict]
                    if current_f1 == max(all_f1s):
                        section += f'$\\underline{{\\mathbf{{{current_f1:.3f}}}}}$ & '
                        gains_acc[k + j * len(TRAINING_MODES)] = sub_results_dict[model]['percentage_f1_gain'] * 100
                    else:
                        section += f'${current_f1:.3f}$ & '
                except KeyError as e:
                    section += '--' + ' & '

        section = section[:-2] + '\\\\' + '\n'

    return section


# endregion

def generate_latex_table(results):
    results = _post_process(results)

    num_cols = 1 + len(TRAINING_MODES) * len(DATASETS)
    gains_acc = [0] * (num_cols - 1)

    backslash = '\\'

    return f'''\\begin{{tabular}}{{{'l' + 'c' * (num_cols - 1)}}}
    \\hline
    \\multirow{{2}}{{*}}{{ Models }} &
    {_datasets_section()}
    
    \\cline {{ 2 - {num_cols} }} &
        {(' Unsup. F1 & Sup. F1 & ' * ((num_cols - 1) // 2))[:-2]} \\\\
    
    \\hline
    {_models_section(results, gains_acc)}
    
    \\hline
    \\% Gain over Raw Features & {' & '.join(f'{gain:.0f}{backslash}%' for gain in gains_acc)} \\\\
    \\hline
    \\end{{tabular}}'''


if __name__ == '__main__':
    results_dir = args.results_dir
    assert osp.exists(results_dir), f'{results_dir} does not exist'
    results = json.load(open(osp.join(results_dir, 'table1.json'), 'r'))
    print(generate_latex_table(results))
