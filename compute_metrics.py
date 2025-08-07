import pandas as pd
import numpy as np
import os
import argparse
import re
import tqdm
from datetime import datetime

def find_most_recent_directory(folder_path, dataset):
    pattern = re.compile(rf"results_{re.escape(dataset)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})")
    current_time = datetime.now()
    closest_file = None
    closest_time_diff = float("inf")
    
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    for folder in folders:
        match = pattern.match(folder)
        if match:
            file_datetime = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
            time_diff = abs((file_datetime - current_time).total_seconds())
            if time_diff < closest_time_diff:
                closest_time_diff = time_diff
                closest_file = folder
    
    return closest_file

def get_normalized_entropies(soft_labels):
    entropies = - np.sum(soft_labels * np.log(soft_labels+1e-7), axis = -1)
    max_entropy = -np.log(1/soft_labels.shape[-1])
    return entropies / max_entropy

def get_CCC(covers, labels):
    classes = np.sort(np.unique(labels))
    CCC_list = []
    for cl in classes:
        mask = labels==cl
        CCC_list.append(np.mean(covers[...,mask], axis=-1))
    return np.stack(CCC_list).T

def get_MCCC(covers, labels):
    ccc = get_CCC(covers, labels)
    return np.min(ccc, axis=-1)

def get_covgap(covers, labels, alphas):
    ccc = get_CCC(covers, labels)
    return np.mean(np.abs(ccc-(1-alphas[:,None])),axis=-1)

def get_qbin_mask(data, n_bins):
    def bin_func(arr):
        quantile_edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
        return np.digitize(arr, bins=quantile_edges[1:-1], right=True)
    
    return np.apply_along_axis(bin_func, axis=-1, arr=data)

def get_ECC(covers, entropies, n_bins):
    qbins = get_qbin_mask(entropies, n_bins)
    ECC_list = []
    for bin in range(n_bins):
        mask = qbins==bin
        masked_covers = np.ma.masked_array(covers, mask=~mask)
        ECC_list.append(np.mean(masked_covers, axis=-1))
    return np.stack(ECC_list).T

def get_entropy_covgap(covers, entropies, alphas, n_bins):
    ecc = get_ECC(covers, entropies, n_bins)
    return np.mean(np.abs(ecc-(1-alphas[:,None])),axis=-1)

def get_MECC(covers, labels, n_bins):
    ecc = get_ECC(covers, labels, n_bins)
    return np.min(ecc, axis=-1)

if __name__ == '__main__':
    if "PATH_TO_RESULTS" in os.environ.keys():
        PATH_TO_RESULTS = os.environ["PATH_TO_RESULTS"]
    else:
        PATH_TO_RESULTS = os.path.join("..", "..", "results")
    parser = argparse.ArgumentParser(description="Replicating the paper figures.")
    parser.add_argument("--dataset", type=str, help="The name of the dataset and model used.")
    parser.add_argument("--conformal_method", type = str, default = 'aps')
    parser.add_argument("--n_folds", default=10, type=int, help="Number of folds to repeat the experiment and obtain less noisy results.")
    parser.add_argument("--backbone", default="ViT-B/16", type=str, help="CLIP backbone used.")
    parser.add_argument("--non_local", action='store_true')
    parser.add_argument("--show", action="store_true", help="If this flag is added, shows the figures on top if saving them.")
    #parser.add_argument("--figures", default=[1,2,3,5], nargs='+', type=int, help="The figures from the article to generate, adapted to one dataset. Only from figures 1, 2, 4, and 5. Other figures are irrelevant with a single dataset or model.")
    args = parser.parse_args()
    n_folds = args.n_folds
    
    
    bname = args.backbone.replace('-', '_').replace('/', '')
    if args.non_local:
        pattern = f"results_{args.dataset}_{args.conformal_method}_{bname}"
    else:
        pattern = f"results_{args.dataset}_local_{args.conformal_method}_{bname}"
    
    #li_matches = [f for f in os.listdir(args.results_dir) if pattern.fullmatch(f)]
    li_matches = [f for f in os.listdir(PATH_TO_RESULTS) if pattern in f]
    assert len(li_matches) <= 2
    
    metrics_dfs_list = []
    set_sizes_df_list = []
    args.results_dir = os.path.join(PATH_TO_RESULTS, li_matches[0])
    

    for fold in tqdm.tqdm(range(n_folds), desc='Loading results...'):
                
        
        data = np.load(os.path.join(args.results_dir,f'results_fold_{fold}.npz'))
        print(data['tau_values'])
        print(data['m_values'])
        #data_sl = np.load(os.path.join(args.results_dir,f'results_fold_{fold}_soft_labels.npz'))

        set_sizes = np.sum(data['y_sets'], axis=-1)
        covers = data['y_sets'][np.arange(data['y_sets'].shape[0])[:, None], np.arange(data['y_sets'].shape[1]), data['labels']]
        #entropies = np.repeat(get_normalized_entropies(data_sl['soft_labels']),len(np.unique(data['alphas'])), axis=0)
        metrics_dfs_list.append(pd.DataFrame({
            'fold':fold,
            'temperature':data['temperatures'],
            'alpha':data['alphas'],
            'set_sizes_mean':np.mean(set_sizes,axis=-1),
            'set_sizes_std':np.std(set_sizes, axis=-1),
            'set_sizes_med':np.median(set_sizes, axis=-1),
            'set_sizes_quant':np.quantile(set_sizes, 0.9, axis=-1),
            'coverage':np.mean(covers,axis=-1),
            'covgap':get_covgap(covers,data['labels'], data['alphas']),
            'MCCC':get_MCCC(covers,data['labels']),
            'tau_values':data['tau_values'],
            'm_values':data['m_values'],

        }))


    metrics_df = pd.concat(metrics_dfs_list).reset_index()
    pruned_metrics_df = metrics_df[metrics_df['alpha'].isin(np.sort(metrics_df['alpha'].unique()[::2]))]
    
    import pickle
    
    rpath = os.path.join(PATH_TO_RESULTS, bname)
    os.makedirs(rpath, exist_ok=True)
    
    if args.non_local:
        suff = ""
    else:
        suff = "_local"
        
    suff = "" if args.non_local else "_local"
    save_dict = {}
    save_dict['metrics'] = metrics_df
    save_dict['params'] = {'tau_values':data['tau_values'],
                'm_values':data['m_values']}
    
    with open(os.path.join(rpath, f"metrics_{args.dataset}{suff}.pickle"), 'wb') as f:
        pickle.dump(save_dict, f)
        

    