import pandas as pd
import numpy as np
import os
import argparse
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
from tqdm import tqdm
cp_path = "/linux/clfuchs/github/cp_euvip/"
if cp_path not in sys.path:
    sys.path.append(cp_path)
  

PLOT_CVG_ENTROPY = True
PLOT_ENTR_VS_SETSIZE = True
PLOT_EXCESSES = True
PLOT_EXCESSES_KDE = True
jtemps_to_plot = [18]
jalphas_to_plot = [1,4,7]
METHOD = 'APS'

datasets=[
  "UCF101",
  "hmdb51",
  #"Kinetics400",
  #"sun397",
  #"Food101",
  "dtd",
  "OxfordPets",
  "eurosat",
  "StanfordCars",
  "Caltech101",
  "Flower102",
  "fgvc_aircraft",
]

backbones=[
  "ViT-B/16",
  "ViT-L/14",
  "ViT-B/32",
  "RN50",
  "RN101",
]
if METHOD == 'LAC':
    base_results_path = "/dir/scratchL2/clfuchs/euvip_results/"
    fig_save_path = "/linux/clfuchs/euvip_figures/"
elif METHOD == 'APS':
    base_results_path = "/linux/clfuchs/euvip_results_APS/"
    fig_save_path = "/linux/clfuchs/euvip_figures_APS/"
  
  
os.makedirs(fig_save_path, exist_ok = True)
li_results = os.listdir(base_results_path)
backbone = backbones[0]
all_results = {}
for dataset_name in datasets:
    bname = backbone.replace('-', '_').replace('/','')
    name_pattern = f"results_{dataset_name}_{bname}"
    li_matched_results = [u for u in li_results if name_pattern in u]
    print(f'Number of matched results : {len(li_matched_results)}')

    all_results[dataset_name] = {}
    
    matched_result = li_matched_results[0]
    num_folds = len(os.listdir(os.path.join(base_results_path, matched_result)))//2 #dirty, because soft labels are not always here; in that case it bugs out later on so it's mostly fine
    for jfold in tqdm(range(num_folds)):
        all_results[dataset_name][f'fold_{jfold}'] = {}
        all_results[dataset_name][f'fold_{jfold}']['cp_results'] =  np.load(os.path.join(base_results_path, matched_result,
                                                                        f"results_fold_{jfold}.npz"))  
        all_results[dataset_name][f'fold_{jfold}']['soft_pred'] =  np.load(os.path.join(base_results_path, matched_result,
                                                                f"results_fold_{jfold}_soft_labels.npz"))  
jfold = 0                                                             
for dataset_name in datasets:
    
    print(f'\n ========== Dataset : {dataset_name}')
    
    n_bins = 10
    ntemps = 39
    nalphas = 8
    h = np.zeros((num_folds, ntemps, n_bins))
    b = np.zeros((num_folds, ntemps, n_bins+1))
    all_covers_per_bin = np.zeros((num_folds, ntemps*nalphas, n_bins))
    all_nsamples_per_bins = np.zeros((num_folds, ntemps, n_bins))
    all_opti_set_sizes_per_entropy_bin = np.zeros((num_folds, ntemps, n_bins))
    
    all_n_excess = None 
    #all_n_lacking = None
    all_folds_all_entropies = None 
    all_folds_all_set_sizes = None
    for jfold in tqdm(range(num_folds)):
        temp_grid = all_results[dataset_name][f'fold_{jfold}']['cp_results']['temperatures']
        alpha_grid = all_results[dataset_name][f'fold_{jfold}']['cp_results']['alphas']
        gt_labels = all_results[dataset_name][f'fold_{jfold}']['cp_results']['labels']
        cp_sets = all_results[dataset_name][f'fold_{jfold}']['cp_results']['y_sets'].astype('bool').squeeze()
        soft_pred = all_results[dataset_name][f'fold_{jfold}']['soft_pred']['soft_labels']
        utemps = np.unique(temp_grid)
        ualphas = np.unique(alpha_grid)
        ntemps = utemps.shape[0]
        nalphas = ualphas.shape[0]
        
        #print(f'Unique alphas : {ualphas}')
        #print(f'Unique temps : {utemps}')
        assert ntemps == utemps.shape[0], utemps.shape[0]
        
        N = soft_pred.shape[1]
        K = soft_pred.shape[2]
        #print('max gt labels',np.max(gt_labels)+1)
        if all_folds_all_entropies is None:
            all_folds_all_entropies = np.zeros((num_folds, ntemps, N))
            all_folds_all_set_sizes = np.zeros((num_folds, ntemps*nalphas, N))
            all_n_excess = np.zeros((num_folds, ntemps*nalphas, N))
            all_opti_set_sizes = np.zeros((num_folds, ntemps, N))
            #all_n_lacking = np.ones((num_folds, ntemps*nalphas, N)) * np.nan
        #print('cp sets shape', cp_sets.shape)
        all_folds_all_set_sizes[jfold,...] = np.sum(cp_sets, axis = -1)
        
        #compute covers
        A = cp_sets.shape[0] #ntemps * nalphas
        B = cp_sets.shape[1] #N
        idx0 = np.arange(A)[:, None]       # shape (A, 1)
        idx1 = np.arange(B)[None, :]       # shape (1, B)
        idx2 = gt_labels[None, :]          # shape (1, B)
        # Use advanced indexing
        is_covered = cp_sets[idx0, idx1, idx2]  # shape (A, B)
        covers = np.sum(is_covered, axis = 1)/is_covered.shape[1]
        covgaps = covers - (1-alpha_grid)
        #print(f'Number of alphas : {nalphas}')
        #print(f'Number of temp : {ntemps}')
        
        # get number of excessive or lacking classes chosen in conformal sets
        soft_labels_order = np.argsort(-soft_pred[0,...], axis = -1) #the order is the same for all temperatures and all alphas 
        #print('soft labels_order shape', soft_labels_order.shape)
        #print('soft pred shape',soft_pred.shape)
        
        mask_opti_set_size = soft_labels_order == gt_labels[:,None] 
        opti_set_size = np.repeat(np.arange(K)[None,:], repeats = N, axis = 0)[mask_opti_set_size].reshape(N)+1 #opti set sizes are the same for all temperatures
        all_opti_set_sizes[jfold, :, :] = opti_set_size[...]
        opti_set_size = np.repeat(opti_set_size[None,:], ntemps*nalphas, axis = 0)
        #print(f'Farthest gt label is at rank : ', np.max(opti_set_size-1))
        #cp_ss = all_folds_all_set_sizes[jfold,...][is_covered]
        excesses = all_folds_all_set_sizes[jfold,...] - opti_set_size
        all_n_excess[jfold,...] = excesses
        
        #s = 19
        #print('is covered shape : ', is_covered.shape)
        #print('cp set size shape :', all_folds_all_set_sizes[jfold,...].shape)
        #print('excesses shape : ', excesses.shape)
        #print('opti set size shape : ', opti_set_size.shape)
        #print('gt label : ', gt_labels[s])
        #print('order soft labels : ', soft_labels_order[s,:])
        #print('soft pred of gt label : ', soft_pred[-2,s,gt_labels[s]])
        #print('ordered soft labels : ', soft_pred[-2,s,soft_labels_order[s,:]])
        #print('opti set size : ', opti_set_size[:10, s])
        #print('cp set size : ', all_folds_all_set_sizes[jfold,:10, s])
        #print('is covered : ', is_covered[:10,s])
        #print('excesses : ', excesses[:10,s])
        #tut =a+b
        #raise RuntimeError
        
        #print(cp_sets.shape)
        #print(gt_labels.shape)
        #print(gt_labels)
        #print()
        entropies = - np.sum(soft_pred * np.log(soft_pred+1e-7), 
                             axis = -1)
        max_entropy = -np.log(1/K)
        entropies = entropies/max_entropy
        all_folds_all_entropies[jfold,...] = entropies
        #h,b = np.histogram(entropies, bins = 10)
        #print("==== Entropies")
        #print(np.min(entropies))
        #print(np.max(entropies))
        #print(np.mean(entropies))
        #print(np.std(entropies))
        li_masks = []
        for jtemp in range(ntemps):
            li_masks.append([])
            h_,b_ = np.histogram(entropies[jtemp,:], bins = n_bins)
            b[jfold, jtemp, :] = b_[...]
            h[jfold, jtemp, :] = h_[...]
            for jbin in range(n_bins):
                li_masks[-1].append(np.logical_and(entropies[jtemp,:]>b_[jbin], entropies[jtemp,:]<= b_[jbin+1]))

        covers_per_mask = []
        
        for jtemp, masks_single_temp in enumerate(li_masks):
            #print(len(masks_single_temp))
            for jbin in range(n_bins):
                mask = masks_single_temp[jbin]
                #print(mask.shape)
                mask_temp = temp_grid == utemps[jtemp]
                masked_cp_sets = cp_sets[mask_temp,:,:][:, mask, :]
                #compute covers
                A = masked_cp_sets.shape[0]
                B = masked_cp_sets.shape[1]
                idx0 = np.arange(A)[:, None]       # shape (A, 1)
                idx1 = np.arange(B)[None, :]       # shape (1, B)
                idx2 = gt_labels[mask][None, :]          # shape (1, B)
                # Use advanced indexing
                is_covered = masked_cp_sets[idx0, idx1, idx2]  # shape (A, B)
                #print(is_covered.shape)
                covers = np.sum(is_covered, axis = 1)/is_covered.shape[1]
                #print(covers)
                all_covers_per_bin[jfold, mask_temp, jbin] = covers[...]
                all_nsamples_per_bins[jfold, jtemp, jbin] = np.sum(mask)
                #print(f'jtemp : {jtemp} | nsamples : {np.sum(mask)}')
                
                
                # get the averae opti set size per entropy bin
                all_opti_set_sizes_per_entropy_bin[jfold,:,jbin] = np.mean(opti_set_size[0,mask])
                
                
    from scipy.interpolate import interp1d
    from matplotlib import cm
    from matplotlib.colors import Normalize

    if PLOT_CVG_ENTROPY:
        # Set up the colormap
        cmap = cm.get_cmap('jet')
        norm = Normalize(vmin=0, vmax=nalphas - 1)
        for jtemp in tqdm(jtemps_to_plot):
    
            b_m_ = .5 * (b[:, jtemp, 1:] + b[:, jtemp, :-1])  # shape: [num_folds, nbins-1]
            fig, axes = plt.subplots(1, 1, squeeze=False)
            ax = axes[0, 0]
            all_nsamples_interp = []
            common_x = None
            for jalpha in jalphas_to_plot:
                all_interp_ys = []
                all_interp_nsamples_per_bin = []
                

                # Determine a common x-grid by pooling all valid x across folds
                if jalpha == jalphas_to_plot[0]:
                    pooled_x = []
                    for jfold in range(num_folds):
                        cover_per_bin = all_covers_per_bin[jfold, jtemp * nalphas + jalpha, :]
                        mask_nan = np.isnan(cover_per_bin)
                        x = b_m_[jfold, ~mask_nan]
                        pooled_x.append(x)
                    pooled_x = np.concatenate(pooled_x)
                    xmin, xmax = np.min(pooled_x), np.max(pooled_x)
                    common_x = np.linspace(xmin, xmax, 100) #this is dumb because the common x is always the same for a fixed temperature, but anyway

                # Interpolate y and nsamples for each fold
                for jfold in range(num_folds):
                    cover_per_bin = all_covers_per_bin[jfold, jtemp * nalphas + jalpha, :]
                    nsamples_per_bin = all_nsamples_per_bins[jfold, jtemp, :]
                    mask_nan = np.isnan(cover_per_bin)
                    x = b_m_[jfold, ~mask_nan]
                    y = cover_per_bin[~mask_nan]
                    if len(x) < 2:
                        continue
                    interp_y = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)(common_x)
                    interp_n = interp1d(x, nsamples_per_bin, kind='linear', bounds_error=False, fill_value=np.nan)(common_x)
                    all_interp_ys.append(interp_y)
                    all_interp_nsamples_per_bin.append(interp_n)

                # Stack and average, ignoring NaNs
                all_interp_ys = np.array(all_interp_ys)
                all_interp_nsamples_per_bin = np.array(all_interp_nsamples_per_bin)
                mean_y = np.nanmean(all_interp_ys, axis=0)
                mean_nsamples = np.nanmean(all_interp_nsamples_per_bin, axis=0)
                all_nsamples_interp.append(mean_nsamples)

                color = cmap(norm(jalpha))
                ax.plot(common_x, mean_y, label=f'$\\alpha = {ualphas[jalpha]}$', color=color)
                ax.plot([common_x[0]*0.95, common_x[-1]*1.05], 
                        [1-ualphas[jalpha], 1-ualphas[jalpha]], 
                        color=color, linestyle='--')
                
            # ==== Add colorbar for sample density ====
            # Stack across alphas and average
            all_nsamples_interp = np.nanmean(np.stack(all_nsamples_interp), axis=0)
            nsample_img = all_nsamples_interp[np.newaxis, :]  # shape (1, len)

            # Create an inset axis for the colorbar below
            cb_ax = fig.add_axes([0.125, 0.01, 0.775, 0.1])  # [left, bottom, width, height]
            im = cb_ax.imshow(nsample_img, aspect='auto', cmap='Greens', extent=[common_x[0], common_x[-1], 0, 1])
            cb_ax.set_yticks([])
            cb_ax.set_xticks([])
            cb_ax.set_xlabel("Bin sample density")
            cb_ax.plot(common_x, mean_nsamples)
            fig.colorbar(im, cax=cb_ax, orientation='horizontal')

            ax.set_title(f'temp : {utemps[jtemp]}')
            ax.legend()
            fig.savefig(os.path.join(fig_save_path, f'{dataset_name}_temp_{utemps[jtemp]}.jpg'))
            plt.close(fig)
            
            
           
                
    #import seaborn as sns    
    from scipy.stats import gaussian_kde

    if PLOT_ENTR_VS_SETSIZE:
        for jtemp in tqdm(jtemps_to_plot):
            for jalpha in jalphas_to_plot:
                jtempalpha = jtemp * nalphas + jalpha
                fig, axes = plt.subplots(1, 1, squeeze=False)

                x = all_folds_all_entropies[:, jtemp, :].flatten()
                y = all_folds_all_set_sizes[:, jtempalpha, :].flatten()

                # Remove NaNs
                valid_mask = ~np.isnan(x) & ~np.isnan(y)
                x = x[valid_mask]
                y = y[valid_mask]
                
                # sub sample to speed up
                max_points = 10000
                if len(x) > max_points:
                    indices = np.random.choice(len(x), size=max_points, replace=False)
                    x = x[indices]
                    y = y[indices]
                # Assume x and y are already filtered
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy, bw_method='scott')  # or try 'silverman'

                # Grid over the data range
                xgrid = np.linspace(np.min(x), np.max(x), 100)
                ygrid = np.linspace(np.min(y), np.max(y), 100)
                X, Y = np.meshgrid(xgrid, ygrid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                # Plot
                ax = axes[0, 0]
                cf = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
                c = ax.contour(X, Y, Z, levels=5, colors='black', linewidths=0.5)
                fig.colorbar(cf, ax=ax)

                fig.savefig(os.path.join(
                    fig_save_path,
                    f'{dataset_name}_temp_{utemps[jtemp]}_alpha_{ualphas[jalpha]}_ss_vs_e.jpg'
                ))
                plt.close(fig)
                
                
    if PLOT_EXCESSES:
        all_mean_set_sizes = np.mean(all_folds_all_set_sizes, axis = 0)
        mean_set_sizes_tempalpha = np.mean(all_mean_set_sizes, axis = 1)
        jtempsalphas_to_plot = [jtemp*nalphas + jalpha for jtemp in jtemps_to_plot for jalpha in jalphas_to_plot]
        from contextlib import redirect_stdout
        mode = 'w' if dataset_name == datasets[0] else 'a'
        with open('output.txt', mode) as f:
            with redirect_stdout(f):
                print(f'\n\n ========================= Dataset : {dataset_name} =========================')
                print('Number of classes : ', K)
                print('Number of samples : ', N)
                print('CP set sizes : ')
                
                for jta in jtempsalphas_to_plot:
                    print(f'Temp : {temp_grid[jta]} | Alpha : {alpha_grid[jta]} | mean ss : {mean_set_sizes_tempalpha[jta]}')
                #for jtemp in tqdm(jtemps_to_plot):
                    
                
                from matplotlib import cm
                for jalpha in jalphas_to_plot:
                    for jtemp in jtemps_to_plot:
                        print(f'\n**** Temp : {temp_grid[jta]} | Alpha : {alpha_grid[jta]}')
                        jta = jtemp * nalphas + jalpha
                        current_excess = all_n_excess[:,jta,:]
                        excess_covered = current_excess[current_excess>=0]
                        print(f'\n===== For covered samples, the set size are on average too big of {np.mean(excess_covered)}')
                        print(f'Max : {np.max(excess_covered)}')
                        print(f'Min (should be zero) : {np.min(excess_covered)}')
                        print(f'Proportion with optimal set size : {np.sum(excess_covered==0)/excess_covered.shape[0]}')
                        
                        lacking_covered = -all_n_excess[all_n_excess<0]
                        print(f'\n===== For non covered samples, the set size are on average too small of {np.mean(lacking_covered)}')
                        print(f'Max : {np.max(lacking_covered)}')
                        print(f'Min : {np.min(lacking_covered)}')
                        
                        
                        optimal_sets_sizes = all_folds_all_set_sizes - all_n_excess 
                        print(f'\n===== For coverage = 1 and optimal sets, the average set size would be {np.mean(optimal_sets_sizes)}')
                        print(f'Max : {np.max(optimal_sets_sizes)}')
                        print(f'Min : {np.min(optimal_sets_sizes)}')
                        print(f'Std : {np.std(optimal_sets_sizes)}')
                        
                        fig, axes = plt.subplots(1, 1, squeeze=False)
                        #h_e, b_e = np.histogram(current_excess.size, bins = )
                        #b_em = .5 * (b_e[:-1] + b_e[1:])
                        axes[0,0].plot(np.sort(current_excess.flatten()))
                        axes[0,0].set_ylabel('Excess Set Size')
                        axes[0,0].set_title(f' Method : {METHOD}')
                        fig.savefig(os.path.join(fig_save_path, f'{dataset_name}_temp_{utemps[jtemp]}_alpha_{ualphas[jalpha]}_excess.jpg'))
                        plt.close(fig)
                        
                        
                        
            
            #all_opti_set_sizes_per_entropy_bin
            if False:
                for jtemp in tqdm(jtemps_to_plot):
                    b_m_ = .5 * (b[:, jtemp, 1:] + b[:, jtemp, :-1])  # shape: [num_folds, nbins-1]
                    fig, axes = plt.subplots(1, 1, squeeze=False)
                    # Determine a common x-grid by pooling all valid x across folds
                    jalpha = 0
                    pooled_x = []
                    all_interp_ys = []
                    for jfold in range(num_folds):
                        cover_per_bin = all_covers_per_bin[jfold, jtemp * nalphas + jalpha, :]
                        mask_nan = np.isnan(cover_per_bin)
                        x = b_m_[jfold, :]
                        pooled_x.append(x)
                    pooled_x = np.concatenate(pooled_x)
                    xmin, xmax = np.min(pooled_x), np.max(pooled_x)
                    common_x = np.linspace(xmin, xmax, 100) #the common x is always the same for a fixed temperature
                        
                    # Interpolate y for each fold
                    for jfold in range(num_folds):
                        x = b_m_[jfold, ~mask_nan]
                        y = all_opti_set_sizes_per_entropy_bin[jfold, jtemp, :].flatten()
                        if len(x) < 2:
                            continue
                        interp_y = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)(common_x)
                        all_interp_ys.append(interp_y)
                    # Stack and average, ignoring NaNs
                    all_interp_ys = np.array(all_interp_ys)
                    mean_y = np.nanmean(all_interp_ys, axis=0)

                    #color = cmap(norm(jalpha))
                    axes[0,0].plot(common_x, mean_y, color='g')
                    axes[0,0].set_title(f'Method : {METHOD} | temp : {utemps[jtemp]}')
                    axes[0,0].set_xlabel('Normalized entropy')
                    axes[0,0].set_ylabel('Mean optimal set size')
                    axes[0,0].legend()
                    fig.savefig(os.path.join(fig_save_path, f'{dataset_name}_temp_{utemps[jtemp]}_opti_set.jpg'))
                    plt.close(fig)
            for jtemp in tqdm(jtemps_to_plot):
                for jalpha in jalphas_to_plot:
                    jtempalpha = jtemp * nalphas + jalpha
                    fig, axes = plt.subplots(1, 1, squeeze=False)

                    
                    y = all_opti_set_sizes[:, jtemp, :].flatten()#all_n_excess[:,jtempalpha,:].flatten()
                    x = all_folds_all_entropies[:, jtemp, :].flatten() #entropies only depend on the temperature
                    # Remove NaNs
                    valid_mask = ~np.isnan(x) & ~np.isnan(y)
                    x = x[valid_mask]
                    y = y[valid_mask]  
                    
                    # sub sample to speed up
                    max_points = 20000
                    if len(x) > max_points:
                        indices = np.random.choice(len(x), size=max_points, replace=False)
                        x = x[indices]
                        y = y[indices]

                    # Assume x and y are already filtered
                    xy = np.vstack([x, y])
                    kde = gaussian_kde(xy, bw_method='scott')  # or try 'silverman'

                    # Grid over the data range
                    xgrid = np.linspace(np.min(x), np.max(x), 100)
                    ygrid = np.linspace(np.min(y), np.max(y), 100)
                    X, Y = np.meshgrid(xgrid, ygrid)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                    # Plot
                    fig, axes = plt.subplots(1, 1, squeeze=False)
                    ax = axes[0, 0]
                    cf = ax.contourf(X, Y, Z, levels=50, cmap='turbo')
                    c = ax.contour(X, Y, Z, levels=5, colors='black', linewidths=0.3)
                    axes[0,0].set_xlabel('Entropy')
                    axes[0,0].set_ylabel('Optimal Set Size')
                    fig.colorbar(cf, ax=ax)

                    fig.savefig(os.path.join(
                        fig_save_path,
                        f'{dataset_name}_temp_{utemps[jtemp]}_alpha_{ualphas[jalpha]}_optiss_vs_e.jpg'
                    ))
                    plt.close(fig)
    if PLOT_EXCESSES_KDE:
        for jtemp in tqdm(jtemps_to_plot):
            
            for jalpha in jalphas_to_plot:
                jtempalpha = jtemp * nalphas + jalpha
                fig, axes = plt.subplots(1, 1, squeeze=False)

                
                y = all_n_excess[:,jtempalpha,:].flatten()
                x = all_folds_all_entropies[:, jtemp, :].flatten() #entropies only depend on the temperature
                # Remove NaNs
                valid_mask = ~np.isnan(x) & ~np.isnan(y)
                x = x[valid_mask]
                y = y[valid_mask]  
                
                # sub sample to speed up
                max_points = 20000
                if len(x) > max_points:
                    indices = np.random.choice(len(x), size=max_points, replace=False)
                    x = x[indices]
                    y = y[indices]

                # Assume x and y are already filtered
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy, bw_method='scott')  # or try 'silverman'

                # Grid over the data range
                xgrid = np.linspace(np.min(x), np.max(x), 100)
                ygrid = np.linspace(np.min(y), np.max(y), 100)
                X, Y = np.meshgrid(xgrid, ygrid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                # Plot
                fig, axes = plt.subplots(1, 1, squeeze=False)
                ax = axes[0, 0]
                cf = ax.contourf(X, Y, Z, levels=50, cmap='turbo')
                c = ax.contour(X, Y, Z, levels=5, colors='black', linewidths=0.75)
                axes[0,0].set_xlabel('Entropy')
                axes[0,0].set_ylabel('Excess Set Size')
                axes[0,0].set_title(f' Method : {METHOD}')
                fig.colorbar(cf, ax=ax)

                fig.savefig(os.path.join(
                    fig_save_path,
                    f'{dataset_name}_temp_{utemps[jtemp]}_alpha_{ualphas[jalpha]}_excess_ss_vs_e.jpg'
                ))
                plt.close(fig)


            
            
            
                
                
            
    
            
    
        
    
