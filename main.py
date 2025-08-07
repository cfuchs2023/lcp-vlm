import os
import argparse
import numpy as np
import tqdm
import clip
import torch
import itertools
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from datasets.loaders import dataloader_dict
from datasets.utils import calib_test_sets_folds, get_soft_labels

import warnings
warnings.filterwarnings("ignore") #dirty dont push this to public
# there is a deprecation warning in scikit learn, no idea which function calls it
# we should look into it

def localized_conformal_prediction(
    ordered_conformal_scores,
    cumulative_proba_matrix,
    localizer_weights,
    alphas
):
    n = len(ordered_conformal_scores)

    # Precompute l_index using searchsorted
    l_index = np.searchsorted(ordered_conformal_scores, ordered_conformal_scores, side='left') - 1
    l_index = np.append(l_index, n) # Pad for indexing up to n

    # Precompute denominators
    denom = localizer_weights[1, :] + cumulative_proba_matrix[:, -1]

    theta_p = (cumulative_proba_matrix[np.arange(cumulative_proba_matrix.shape[0]),l_index] + localizer_weights[1,:]) / denom
    theta = cumulative_proba_matrix[np.arange(cumulative_proba_matrix.shape[0]),l_index] / denom
    theta_tilde = np.cumsum(localizer_weights[0,:])[l_index] / np.sum(localizer_weights[0,:])

    theta[l_index==-1] = 0
    theta_tilde[l_index==-1] = 0
    theta_p[l_index==-1] = localizer_weights[1,l_index==-1] / denom[l_index==-1]

    # Define A_1, A_2, A_3 using vectorized operations
    a_1 = np.where(theta_p < theta_tilde)[0]
    a_2 = np.where(theta >= theta_tilde)[0]
    a_3 = np.where((theta_p >= theta_tilde) & (theta < theta_tilde))[0]

    theta_A1 = np.sort(theta_p[a_1])
    theta_A2 = np.sort(theta[a_2])
    theta_A3 = np.sort(l_index[a_3])

    c_1 = np.searchsorted(theta_A1, theta_tilde, side='left')
    c_2 = np.searchsorted(theta_A2, theta_tilde, side='left')
    c_3 = np.searchsorted(theta_A3, l_index, side='left')
    S_k = (c_1+c_2+c_3)/(n+1)

    if isinstance(alphas,np.ndarray) or isinstance(alphas, list):
        k_stars = np.array([np.max(np.argwhere(S_k==np.max(S_k[S_k < 1-alpha])).flatten()) for alpha in alphas])
    else:
        k_stars = np.max(np.argwhere(S_k==np.max(S_k[S_k < 1-alphas])).flatten())
    score_thresh = ordered_conformal_scores[k_stars]

    return score_thresh

def get_conformal_scores(soft_labels, labels, method='lac', evaluate=False, seed=0, 
                         raps_folds=5, alpha_raps=None, k_reg_raps=None, lambda_star_raps=None):
    rng = np.random.default_rng(seed)
    if method == 'lac':
        if evaluate:
            return 1 - soft_labels
        else:
            return 1 - soft_labels[np.arange(len(labels)),labels]
       
    if method == 'aps':
        ordering = np.fliplr(np.argsort(soft_labels, axis=1))
 
        conformal_scores = np.cumsum(np.take_along_axis(soft_labels, ordering, axis=1), axis=1)
 
        reverse_order = np.empty_like(ordering)
        np.put_along_axis(reverse_order, ordering, np.arange(ordering.shape[1]), axis=1)
 
        conformal_scores = np.take_along_axis(conformal_scores, reverse_order, axis=1)
       
        if evaluate:
            return conformal_scores
        else:
            true_label_probs = np.take_along_axis(soft_labels, labels[:, np.newaxis], axis=1)
            u = rng.uniform(size=(len(labels),1))
            return (conformal_scores- u * true_label_probs)[np.arange(len(labels)),labels]
 
    if method == 'topk':
        rank = np.argsort(np.fliplr(np.argsort(soft_labels, axis=1)))
 
        if evaluate:
            return rank
        else:
            return rank[np.arange(len(labels)),labels]
   
    if method == 'raps':
        if evaluate:
            ranks = get_conformal_scores(soft_labels, labels, method='topk', evaluate=True)
            aps_scores = get_conformal_scores(soft_labels, labels, method='aps', evaluate=True)
            return aps_scores + lambda_star_raps * np.maximum(0, ranks-k_reg_raps)
 
        else:
            skf = StratifiedKFold(n_splits=raps_folds, shuffle=True, random_state=seed)
            k_regs = np.empty(raps_folds)
            lambda_stars = np.empty(raps_folds)
            for fold, (train_index, val_index) in enumerate(skf.split(soft_labels, labels)):
                soft_labels_calib, soft_labels_val = soft_labels[train_index], soft_labels[val_index]
                labels_calib, labels_val = labels[train_index], labels[val_index]
                k_regs[fold] = get_k_star(soft_labels_calib, labels_calib, alpha_raps)
                lambda_stars[fold] = get_lambda_star(soft_labels_calib, soft_labels_val, labels_calib, labels_val, k_regs[fold], alpha_raps, rng)
           
            #mode of k_regs
            values, counts = np.unique(k_regs, return_counts=True)
            k_reg = values[np.argmax(counts)]
 
            lambda_star = np.mean(lambda_stars)
 
            ranks = get_conformal_scores(soft_labels, labels, method='topk')
            aps_scores = get_conformal_scores(soft_labels, labels, method='aps', evaluate=True)[np.arange(len(labels)),labels]
            raps_scores = aps_scores + lambda_star * np.maximum(0, ranks-k_reg)
 
            true_label_probs = np.take_along_axis(soft_labels, labels[:, np.newaxis], axis=1)
            u = rng.uniform(size=(len(labels),1))
            return (raps_scores - u * true_label_probs)[np.arange(len(labels)),labels], k_reg, lambda_star
        
        
def get_k_star(soft_labels, labels, alpha):
    ranks = get_conformal_scores(soft_labels, labels, method='topk')
    q_level = np.ceil((len(ranks))*(1-alpha))/len(ranks)
    k_star = np.quantile(ranks, q_level, method='higher')
    return k_star

def get_lambda_star(soft_labels_train, soft_labels_test, labels_train, labels_test, k_reg, alpha, rng):
    lambdas = [.001, .01, .1, .2, .5]
    ranks_train = get_conformal_scores(soft_labels_train, labels_train, method='topk')
    aps_scores_train = get_conformal_scores(soft_labels_train, labels_train, method='aps', evaluate=True)[np.arange(len(labels_train)),labels_train]
    ranks_test = get_conformal_scores(soft_labels_test, labels_test, method='topk', evaluate=True)
    aps_scores_test = get_conformal_scores(soft_labels_test, labels_test, method='aps', evaluate=True)
    true_label_probs = np.take_along_axis(soft_labels_train, labels_train[:, np.newaxis], axis=1)
    u = rng.uniform(size=(len(labels_train),1))
    mean_set_sizes = np.zeros_like(lambdas)
    for l, lamb in enumerate(lambdas):
        raps_scores_train = aps_scores_train + lamb * np.maximum(0, ranks_train-k_reg)
        raps_scores_train = (raps_scores_train - u * true_label_probs)[np.arange(len(labels_train)),labels_train]
        q_level = np.ceil((len(raps_scores_train))*(1-alpha))/len(raps_scores_train)
        q_hat = np.quantile(raps_scores_train, q_level, method='higher')
 
        raps_scores_test = aps_scores_test + lamb * np.maximum(0, ranks_test-k_reg)
        mean_set_sizes[l] = (raps_scores_test <= q_hat).sum(axis=-1).mean()
    return lambdas[np.argmin(mean_set_sizes)]

def apply_sigmoid(S, m, tau, transform_type = 'sigmoid'):
    if transform_type == 'sigmoid':
        a = -m * (1 - tau)
        b = -m * (S - tau)
        out = (1 + np.exp(a)) / (1 + np.exp(np.minimum(b,700)))
    elif transform_type == 'beta':
        # print('beta', m)
        if np.abs(m)>1e-4:
            a = -m * (1-S)
            out = (np.exp(a) - np.exp(-m))/(1 - np.exp(-m))
        else:
            out = S
    elif transform_type == 'identity':
        out = S
    return out

def get_opti_params(H_matrix_base, H_end_stack, train_conformal_scores, 
                    test_soft_labels, test_labels, test_conformal_scores, 
                    m_grid, tau_grid, conformal_method, select_last, alpha,
                    transform_type = 'sigmoid'):
    performances = np.empty((len(m_grid), len(tau_grid)))

    Q_matrix = np.empty((len(train_conformal_scores)+1,len(train_conformal_scores)+1))
    H_end = np.ones((2,len(train_conformal_scores)+1))

    for i, m in enumerate(m_grid):
        for j, tau in enumerate(tau_grid):
            H_matrix_weighted = apply_sigmoid(H_matrix_base, m, tau, transform_type = transform_type)
            H_end_stack_weighted = apply_sigmoid(H_end_stack, m, tau, transform_type = transform_type)
            thresholds = get_local_thresholds(H_end, Q_matrix, train_conformal_scores, test_labels, H_matrix_weighted, H_end_stack_weighted, alpha)
            y_sets = get_conformal_sets(test_conformal_scores, thresholds, test_soft_labels, select_last, conformal_method)
            performances[i,j] = np.mean(np.sum(y_sets,axis=1))
    m_argmin, tau_argmin = np.unravel_index(np.argmin(performances), performances.shape)
    return m_grid[m_argmin], tau_grid[tau_argmin]

def optimize_distance_scoring(m_grid, tau_grid, H_matrix_base, H_end_stack, soft_labels, 
                              labels, conformal_method, select_last, alpha, n_folds=5, seed=0,
                              transform_type = 'sigmoid'):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    m_stars = np.empty(n_folds)
    tau_stars = np.empty(n_folds)
    for fold, (train_index, val_index) in enumerate(skf.split(H_matrix_base, labels)):
        H_submatrix = H_matrix_base[train_index][:,train_index]
        H_substack = H_end_stack[:,train_index]
        
        train_soft_labels = soft_labels[train_index]
        train_labels = labels[train_index]
        test_soft_labels = soft_labels[val_index]
        test_labels = labels[val_index]

        if conformal_method=='raps':
            train_conformal_scores, k_reg, lambda_star = get_conformal_scores(train_soft_labels, train_labels, method=conformal_method, alpha_raps=alpha)
            test_conformal_scores= get_conformal_scores(test_soft_labels, test_labels,method=conformal_method, evaluate=True, lambda_star_raps=lambda_star, k_reg_raps=k_reg, alpha_raps=alpha)
        else:
            train_conformal_scores = get_conformal_scores(train_soft_labels, train_labels, method=conformal_method)
            test_conformal_scores = get_conformal_scores(test_soft_labels, test_labels,method=conformal_method, evaluate=True)

        m_stars[fold], tau_stars[fold] = get_opti_params(
            H_submatrix, 
            H_substack, 
            train_conformal_scores, 
            test_soft_labels, 
            test_labels,
            test_conformal_scores, 
            m_grid, 
            tau_grid, 
            conformal_method, 
            select_last, 
            alpha,
            transform_type = transform_type
        )
    #print(m_stars)
    return np.median(m_stars), np.mean(tau_stars)

def get_conformal_sets(test_conformal_scores, quantiles_ta, test_soft_labels, select_last, conformal_method):
    mask = (test_conformal_scores <= quantiles_ta[:,None])

    first_unselected = np.argmin(np.ma.array(test_conformal_scores, mask=mask), axis=1)
    if conformal_method in ['aps', 'raps'] and select_last == 'True':
        mask[np.arange(mask.shape[0]),first_unselected] = True
    elif conformal_method in ['aps', 'raps'] and select_last == 'randomized':
        v_param = (test_conformal_scores[np.arange(mask.shape[0]),first_unselected]-quantiles_ta)/test_soft_labels[np.arange(mask.shape[0]),first_unselected]
        rng = np.random.default_rng(0)
        u_param = rng.uniform(size=len(v_param))
        random_mask = (v_param - u_param) <= 1e-8
        mask[np.arange(mask.shape[0])[random_mask],first_unselected[random_mask]] = True
    
    return mask

def get_local_thresholds(H_end, Q_matrix, train_conformal_scores, test_labels, H_matrix_weighted, H_end_stack_weighted, alpha):
    ordering = np.argsort(train_conformal_scores)
    Q_matrix[:-1,:-1] = np.cumsum(H_matrix_weighted[ordering,:][:,ordering], axis=1)

    thresholds = np.empty(len(test_labels))
    for test_idx in range(len(test_labels)):
        H_end[:,:-1] = H_end_stack_weighted[test_idx][ordering]

        Q_matrix[-1,:] = np.cumsum(H_end[0])
        Q_matrix[:,-1] = Q_matrix[:,-2].T+H_end[0]

        thresholds[test_idx] = localized_conformal_prediction(train_conformal_scores[ordering], Q_matrix, H_end, alpha)
    return thresholds

def parse_range(range_str):
    """Parses a string of the format 'min:max:step' into a list of values."""
    try:
        min_val, max_val, step = map(float, range_str.split(':'))
        scale = 10 ** (-np.floor(np.log10(step)).astype(int))
        return np.arange(round(min_val * scale), round(max_val * scale) + 1, round(step * scale)) / scale
    except ValueError:
        raise argparse.ArgumentTypeError("Range must be in the format 'min:max:step' with numeric values.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replicating the paper experiments.")
    parser.add_argument("--dataset", default="dtd", type=str, 
                        help="Dataset used. Must be located in 'data' folder and have a corresponding AbstractDataloader object to handle it.")
    parser.add_argument("--backbone", default="ViT-B/16", type=str, 
                        help="CLIP backbone used.")
    parser.add_argument("--n_folds", default=10, type=int, 
                        help="Number of folds to repeat the experiment and obtain less noisy results.")
    parser.add_argument("--n_samples_calib", default=1000, type=int, 
                        help="Number of shots for each label in the calibration set. Uniformly drawn from the training set.")
    parser.add_argument("--m_grid", default=None, type=parse_range, 
                        help='Grid over which to search for the m parameter of the sigmoid. Defaults internally to the values used for the paper.')
    parser.add_argument("--tau_grid", default=None, type=parse_range, 
                        help='Grid over which to search for the tau parameter of the sigmoid. Defaults internally to the values used for the paper.')
    parser.add_argument("--drop_soft_labels", action='store_true', 
                        help="Saves the soft labels if set to True, this is more costly in time and space.")
    parser.add_argument("--conformal_method", default='aps', type=str, 
                        help='Method to use for conformal prediction. Can be "lac", "aps", "raps" or "entropy_raps".')
    parser.add_argument('--select_last', default = 'True', type = str, 
                        help='Whether to always select the last class or not.')
    parser.add_argument('--non_local', action='store_true', 
                        help='Use this parameter to run the non local baselines.')
    
    args = parser.parse_args()
    args.alphas_grid = np.array([0.1])
    args.temperatures_grid = np.array([100]) #1/T
    args.grid_search = True
    args.sim_transform = 'sigmoid'
    if 'PATH_TO_DATASETS' in os.environ.keys():   
        PATH_TO_DATASETS = os.environ["PATH_TO_DATASETS"]
    else:
        PATH_TO_DATASETS = "E:/DATA"
    
    if 'PATH_TO_RESULTS' in os.environ.keys():   
        PATH_TO_RESULTS = os.environ["PATH_TO_RESULTS"]
    else:
        PATH_TO_RESULTS = "E:/DATA/euvip_results"
    
    if 'PATH_TO_FIGURES' in os.environ.keys():   
        PATH_TO_FIGURES = os.environ["PATH_TO_FIGURES"]
    else:
        PATH_TO_FIGURES = "E:/DATA/euvip_results"

    if args.dataset not in dataloader_dict.keys():
        raise ValueError(
            f"""The submitted dataset '{args.dataset}' is not implemented. 
            Please implement a child class of AbstractDataloader to handle 
            your dataset and add it to 'dataloader_dict'"""
        )

    dataset = dataloader_dict[args.dataset](PATH_TO_DATASETS, args.dataset)
    all_features, all_labels = dataset.load_features_and_labels(args.backbone)[:2]
    if len(all_features.shape)==2:
        all_features = all_features[:,None,:]

    train_features_folds, train_labels_folds, \
    test_features_folds, test_labels_folds = calib_test_sets_folds(all_features,
                                                                   all_labels, 
                                                                   total_n_samples=args.n_samples_calib, 
                                                                   n_folds=args.n_folds,
                                                                   )
    K = torch.max(all_labels)+1
    text_embeddings=dataset.get_textual_prototypes(clip.load(args.backbone)[0], model_dim = all_features.shape[-1])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bname = args.backbone.replace('/', '').replace('-', '_')
    suff = "" if args.non_local else "_local"
    output_folder = os.path.join(PATH_TO_RESULTS, f"results_{args.dataset}{suff}_{args.conformal_method}_{bname}_{timestamp}")
    
    os.makedirs(output_folder)
    

    # parameters for file dump
    if args.grid_search:
        if args.sim_transform == 'sigmoid':
            if args.m_grid is None:
                args.m_grid = np.array([1,5,10,15,20,25])
            if args.tau_grid is None:
                args.tau_grid = np.array([0,0.25,0.5,0.75,1])
                
    temperatures, alphas = np.array(list(itertools.product(args.temperatures_grid, args.alphas_grid))).T
    
    if not(args.non_local):
        print('transform_type : ', args.sim_transform)
        print('m grid : ', args.m_grid)
        print('tau grid : ', args.tau_grid)
        print('alpha grid : ', args.alphas_grid)
    
    for fold, (train_features, train_labels, test_features, test_labels) in \
        enumerate(zip(train_features_folds, train_labels_folds, test_features_folds, test_labels_folds)):
        y_sets = []
        quantiles = []
        if not args.drop_soft_labels:
            soft_labels=[]

        if args.non_local:
            H_matrix_base = np.ones((train_features.shape[0], train_features.shape[0]))
            H_end_stack = np.ones((test_features.shape[0], train_features.shape[0]))            

        else:
            H_matrix_base = np.dot(train_features,train_features.T)
            H_end_stack = np.dot(test_features, train_features.T)

        Q_matrix = np.empty((len(train_labels)+1,len(train_labels)+1))
        H_end = np.ones((2,len(train_labels)+1))

        n_ops = len(args.temperatures_grid)*len(args.alphas_grid)*len(args.m_grid)*len(args.tau_grid) if args.grid_search else len(args.temperatures_grid)*len(args.alphas_grid)

        pbar = tqdm.tqdm(total=n_ops, desc=f"Fold n°{fold+1}/{args.n_folds}")
        for temperature in args.temperatures_grid:
            train_soft_labels = get_soft_labels(temperature, train_features, text_embeddings).cpu().numpy()
            test_soft_labels = get_soft_labels(temperature, test_features, text_embeddings).cpu().numpy()
            if not args.drop_soft_labels:
                soft_labels.append(test_soft_labels)
            
            quantiles_t = np.empty((len(args.alphas_grid),len(test_labels)))
            test_conformal_scores_alphas = np.empty((len(args.alphas_grid), *test_soft_labels.shape))
            for alpha in args.alphas_grid:
                if args.conformal_method=='raps':
                    train_conformal_scores, k_reg, lambda_star = get_conformal_scores(train_soft_labels, train_labels.cpu().numpy(), method=args.conformal_method, alpha_raps=alpha)
                    test_conformal_scores= get_conformal_scores(test_soft_labels, test_labels.cpu().numpy(),method=args.conformal_method, evaluate=True, lambda_star_raps=lambda_star, k_reg_raps=k_reg, alpha_raps=alpha)
                else:
                    train_conformal_scores = get_conformal_scores(train_soft_labels, train_labels.cpu().numpy(), method=args.conformal_method)
                    test_conformal_scores = get_conformal_scores(test_soft_labels, test_labels.cpu().numpy(),method=args.conformal_method, evaluate=True)

                if args.grid_search:
                    m_list = args.m_grid
                    tau_list = args.tau_grid

                else:
                    m_star, tau_star = optimize_distance_scoring(args.m_grid, args.tau_grid, H_matrix_base, 
                                                                 H_end_stack, train_soft_labels, 
                                                                 train_labels.cpu().numpy(), 
                                                                 args.conformal_method, args.select_last, alpha,
                                                                 transform_type = args.sim_transform)
                    m_list, tau_list = [m_star],[tau_star]
                    m_values, tau_values = m_list, tau_list

                for m in m_list:
                    for tau in tau_list:
                        H_matrix_weighted = apply_sigmoid(H_matrix_base, m, tau, transform_type = args.sim_transform)
                        H_end_stack_weighted = apply_sigmoid(H_end_stack, m, tau, transform_type = args.sim_transform)

                        quantiles_ta = get_local_thresholds(H_end, Q_matrix, train_conformal_scores, test_labels, H_matrix_weighted, H_end_stack_weighted, alpha)
                    
                        mask = get_conformal_sets(test_conformal_scores, quantiles_ta, test_soft_labels, args.select_last, args.conformal_method)
                        y_sets.append(mask)
                        quantiles.append(quantiles_ta)
                        pbar.update(1)
        pbar.close()

        y_sets = np.stack(y_sets)
        print(y_sets.shape)
        print('mean set sizes per alpha : ', np.mean(np.sum(y_sets, axis = -1), axis = 1))
        print("Saving .npz file...", end="", flush=True)
        dump_path = os.path.join(output_folder,f'results_fold_{fold}.npz')
        np.savez_compressed(
            dump_path, 
            temperatures=temperatures, 
            alphas=alphas,
            m_values=m_values,
            tau_values=tau_values,
            quantiles=np.stack(quantiles), 
            y_sets=y_sets,
            labels = test_labels
        )
        
        if not args.drop_soft_labels:
            dump_path = os.path.join(output_folder,f'results_fold_{fold}_soft_labels.npz')
            np.savez_compressed(
                dump_path, 
                temperatures=args.temperatures_grid, 
                soft_labels=np.stack(soft_labels),
                labels = test_labels
            )
        print(" Done", flush=True)
