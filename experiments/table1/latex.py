import json
import os.path as osp

from icecream import ic

from experiments.table1.settings import TRAINING_MODES, DATASETS, MODELS
from graphsage.settings import args


def _datasets_section():
    section = ''

    for i, dataset in enumerate(DATASETS):
        section += f'\t\\multicolumn{{2}}{{c}}{{ {dataset.upper()} }}'

        if i < len(DATASETS) - 1:
            section += ' & '
        else:
            section += ' \\\\' + '\n'

    return section


def _models_section(results):
    section = ''

    for i, model in enumerate(MODELS):
        section += f"\t{'-'.join(w.upper() for w in model.split('_'))} & "
        for dataset in DATASETS:
            for training_mode in TRAINING_MODES:
                try:
                    current_f1 = results[dataset][training_mode][model]['test_f1']
                    sub_results_dict = results[dataset][training_mode]
                    all_f1s = [sub_results_dict[model]['test_f1'] for model in sub_results_dict]
                    if current_f1 == max(all_f1s):
                        section += f'$\\underline{{\\mathbf{{{current_f1:.3f}}}}}$ & '
                    else:
                        section += f'${current_f1:.3f}$ & '
                except KeyError as e:
                    section += '--' + ' & '

        section = section[:-2] + '\\\\' + '\n'

    return section


def generate_latex_table(results: dict):

    num_cols = 1 + 2 * len(DATASETS)

    return f'''\\begin{{tabular}}{{{'c' * num_cols}}}
    \\hline
    \\multirow{{2}}{{*}}{{ Models }} &
    {_datasets_section()}
    
    \\cline {{ 2 - {num_cols} }} &
        {(' Unsup. F1 & Sup. F1 & '*((num_cols - 1)//2))[:-2]} \\\\
    
    \\hline
    {_models_section(results)}
    
    \\hline
    \\end{{tabular}}'''


if __name__ == '__main__':
    results_dir = args.results_dir
    assert osp.exists(results_dir), f'{results_dir} does not exist'
    results = json.load(open(osp.join(results_dir, 'table1.json'), 'r'))
    print(generate_latex_table(results))
