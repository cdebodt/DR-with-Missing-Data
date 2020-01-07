#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

#
# %%%% !!! IMPORTANT NOTE !!! %%%%
# At the end of this file, a demo presents how this python code can be used. Running this file (python dr_missing_data.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes.
# %%%% !!!                !!! %%%%

#     dr_missing_data.py

# This python code implements a framework to deal with missing data in dimensionality reduction (DR).
# The methodology which is implemented is described in the article "Nonlinear Dimensionality Reduction with Missing Data using Parametric Multiple Imputations", from Cyril de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, published in IEEE Transactions on Neural Networks and Learning Systems, in 2019. 
# Link to retrieve the article: https://ieeexplore.ieee.org/abstract/document/8447227
# At the end of this file, a demo presents how this python code can be used. Running this file (python dr_missing_data.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. The tested versions of the imported packages are specified at the end of the header. 

# If you use this code or the article, please cite as: 
# - de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2019). Nonlinear Dimensionality Reduction With Missing Data Using Parametric Multiple Imputations. IEEE transactions on neural networks and learning systems, 30(4), 1166-1179.
# - BibTeX entry:
#@article{cdb2019drnap,
#  title={{N}onlinear {D}imensionality {R}eduction with {M}issing {D}ata using {P}arametric {M}ultiple {I}mputations},
#  author={de Bodt, C. and Mulders, D. and Verleysen, M. and Lee, J. A.},
#  journal={{IEEE} Trans. Neural Netw. Learn. Syst.},
#  volume={30},
#  number={4},
#  pages={1166--1179},
#  year={2019}
#}

# The main functions of this file are:
# - 'mssne_implem': nonlinear dimensionality reduction through multi-scale SNE (Ms SNE), as presented in the reference [2] below and summarized in [1]. This function enables reducing the dimension of a complete data set. 
# - 'gmm_fit_K', 'gmm_sev_em_fitting' and 'gmm_sev_sampling': Gaussian mixture modeling of a complete or incomplete data set, as presented in [7, 8, 9] and summarized in [1]. These functions respectively enable to:
# ---> 'gmm_fit_K': fit a Gaussian mixture model on a complete or incomplete data set. The number of mixture components is automatically determined and tuned as detailed in [1]. 
# ---> 'gmm_sev_em_fitting': fit a Gaussian mixture model with K components on a complete or incomplete data set. The number K of components is a parameter of the function. Setting it to 1 fits a single multivariate Gaussian on the data set, while setting K to 2 fits two Gaussian components on the data set, etc. 
# ---> 'gmm_sev_sampling': draw samples from a Gaussian mixture model. 
# - 'icknni_implem': implementation of the ICkNNI method as proposed in [15] and employed in the experiments of [1] for the comparison of the performances of the methods. This function enables performing a single imputation of the missing entries in a data set. 
# - 'mssne_na_mmg': nonlinear dimensionality reduction through multi-scale SNE of an incomplete data set, using the methodology presented in [1]. This function enables applying multi-scale SNE on a database with missing values by first fitting a Gaussian mixture model on the data set and then dealing with the missing entries either thanks to multiple imputations or conditional mean imputation. 
# - 'eval_dr_quality': unsupervised evaluation of the quality of a low-dimensional embedding, as introduced in [3, 4] and employed and summarized in [1, 2, 5]. This function enables computing quality assessment criteria measuring the neighborhood preservation from the high-dimensional space to the low-dimensional one. The documentation of the function explains the meaning of the criteria and how to interpret them.
# - 'knngain': supervised evaluation of the quality of a low-dimensional embedding, as introduced in [6]. This function enables computing criteria related to the accuracy of a KNN classifier in the low-dimensional space. The documentation of the function explains the meaning of the criteria and how to interpret them.
# - 'viz_2d_emb' and 'viz_qa': visualization of a 2-D embedding and of the quality criteria. These functions respectively enable to: 
# ---> 'viz_2d_emb': plot a 2-D embedding. 
# ---> 'viz_qa': depict the quality criteria computed by 'eval_dr_quality' and 'knngain'.
# The documentations of the functions describe their parameters. The demo shows how they can be used. 

# Notations:
# - DR: dimensionality reduction.
# - HD: high-dimensional.
# - LD: low-dimensional.
# - HDS: HD space.
# - LDS: LD space.
# - NA: Not Available, synonym of missing data, missing values and missing entry.
# - NAN: Not A Number, synonym of missing data, missing values and missing entry.
# - Ms SNE: multi-scale stochastic neighbor embedding.

# References:
# [1] de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2019). Nonlinear Dimensionality Reduction With Missing Data Using Parametric Multiple Imputations. IEEE transactions on neural networks and learning systems, 30(4), 1166-1179.
# [2] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
# [3] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
# [4] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.
# [5] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.
# [6] de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).
# [7] Eirola, E., Doquire, G., Verleysen, M., & Lendasse, A. (2013). Distance estimation in numerical data sets with missing values. Information Sciences, 240, 115-128.
# [8] Eirola, E., Lendasse, A., Vandewalle, V., & Biernacki, C. (2014). Mixture of gaussians for distance estimation with missing data. Neurocomputing, 131, 32-42.
# [9] Sovilj, D., Eirola, E., Miche, Y., Björk, K. M., Nian, R., Akusok, A., & Lendasse, A. (2016). Extreme learning machine for missing data using multiple imputations. Neurocomputing, 174, 220-231.
# [10] Bouveyron, C., Girard, S., & Schmid, C. (2007). High-dimensional data clustering. Computational Statistics & Data Analysis, 52(1), 502-519.
# [11] Ghahramani, Z., & Jordan, M. I. (1995). Learning from incomplete data.
# [12] Rubin, D. B. (2004). Multiple imputation for nonresponse in surveys (Vol. 81). John Wiley & Sons.
# [13] Little, R. J., & Rubin, D. B. (2014). Statistical analysis with missing data. John Wiley & Sons.
# [14] Cattell, R. B. (1966). The scree test for the number of factors. Multivariate behavioral research, 1(2), 245-276.
# [15] Van Hulse, J., & Khoshgoftaar, T. M. (2014). Incomplete-case nearest neighbor imputation in software measurement data. Information Sciences, 259, 596-610.

# author: Cyril de Bodt (ICTEAM - UCLouvain)
# @email: cyril __dot__ debodt __at__ uclouvain.be
# Last modification date: January 7th, 2020
# Copyright (c) 2020 Universite catholique de Louvain (UCLouvain), ICTEAM. All rights reserved.

# This code was tested with Python 3.7.5 (Anaconda distribution, Continuum Analytics, Inc.). It uses the following modules:
# - numpy: version 1.17.4 tested
# - numba: version 0.46.0 tested
# - scipy: version 1.3.2 tested
# - matplotlib: version 3.1.1 tested
# - scikit-learn: version 0.22 tested
# - pandas: version 0.25.3 tested

# You can use, modify and redistribute this software freely, but not for commercial purposes. 
# The use of this software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.

########################################################################################################
########################################################################################################

import numpy as np, numba, sklearn.decomposition, scipy.spatial.distance, matplotlib.pyplot as plt, scipy.optimize, time, os, pandas, scipy.stats.mstats, scipy.special

# Name of this file
module_name = "dr_missing_data.py"

##############################
############################## 
# General functions used by others in the code. 
####################

@numba.jit(nopython=True)
def close_to_zero(v):
    """
    Check whether v is close to zero or not.
    In:
    - v: a scalar or numpy array.
    Out:
    A boolean or numpy array of boolean of the same shape as v, with True when the entry is close to 0 and False otherwise.
    """
    return np.absolute(v) <= 10.0**(-8.0)

@numba.jit(nopython=True)
def arange_except_i(N, i):
    """
    Create a 1-D numpy array of integers from 0 to N-1 with step 1, except i.
    In:
    - N: a strictly positive integer.
    - i: a positive integer which is strictly smaller than N.
    Out:
    A 1-D numpy array of integers from 0 to N-1 with step 1, except i.
    """
    arr = np.arange(N)
    return np.hstack((arr[:i], arr[i+1:]))

@numba.jit(nopython=True)
def fill_diago(M, v):
    """
    Replace the elements on the diagonal of a square matrix M with some value v.
    In:
    - M: a 2-D numpy array storing a square matrix.
    - v: some value.
    Out:
    M, but in which the diagonal elements have been replaced with v.
    """
    for i in range(M.shape[0]):
        M[i,i] = v
    return M

@numba.jit(nopython=True)
def contains_ident_ex(X):
    """
    Returns True if the data set contains two identical samples, False otherwise.
    In:
    - X: a 2-D numpy array with one example per row and one feature per column.
    Out:
    A boolean being True if and only if X contains two identical rows.
    """
    # Number of samples and of features
    N, M = X.shape
    # Tolerance
    atol = 10.0**(-8.0)
    # For each sample
    for i in range(N):
        if np.any(np.absolute(np.dot((np.absolute(X[i,:]-X[i+1:,:]) > atol).astype(np.float64), np.ones(shape=M, dtype=np.float64)))<=atol):
            return True
    return False

def eucl_dist_matr(X):
    """
    Compute the pairwise Euclidean distances in a data set. 
    In:
    - X: a 2-D np.ndarray with shape (N,M) containing one example per row and one feature per column.
    Out:
    A 2-D np.ndarray dm with shape (N,N) containing the pairwise Euclidean distances between the data points in X, such that dm[i,j] stores the Euclidean distance between X[i,:] and X[j,:].
    """
    return scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X, metric='euclidean'), force='tomatrix')

##############################
############################## 
# Nonlinear dimensionality reduction through multi-scale SNE (Ms SNE) [2]. 
# The main function is 'mssne_implem'. 
# See its documentation for details. 
# The demo at the end of this file and the 'mssne_na_mmg' function present how to use it. 
####################

# Default random seed for Ms SNE. Only used if seed_mssne is set to None in mssne_implem and mssne_sev_implem.
seed_MsSNE_def = 40
# Maximum number of iterations in L-BFGS. It has been chosen large enough to ensure that it is almost always possible for the optimization algorithms to find a solution with a gradient which has a close to zero norm before the maximum number of iterations is reached. 
dr_nitmax = 100000
# The iterations of L-BFGS stop when max{|g_i | i = 1, ..., n} <= gtol where g_i is the i-th component of the gradient. 
dr_gtol = 10**(-5)
# The iterations of L-BFGS stop when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
dr_ftol = 2.2204460492503131e-09
# Maximum number of line search steps per L-BFGS-B iteration.
dr_maxls = 50
# The maximum number of variable metric corrections used to define the limited memory matrix of L-BFGS.
dr_maxcor = 10

n_eps_np_float64 = np.finfo(dtype=np.float64).eps

@numba.jit(nopython=True)
def ms_perplexities(N, K_star=2, L_min=-1, L_max=-1):
    """
    Define exponentially growing multi-scale perplexities, as in [2].
    In:
    - N: number of data points.
    - K_star: K_{*} as defined in [2], to set the multi-scale perplexities.
    - L_min: if -1, set as in [2]. 
    - L_max: if -1, set as in [2]. 
    Out:
    A tuple with:
    - L: as defined in [2]
    - K_h: 1-D numpy array, with the perplexities in increasing order.
    """
    if L_min == -1:
        L_min = 1
    if L_max == -1:
        L_max = int(round(np.log2(np.float64(N)/np.float64(K_star))))
    L = L_max-L_min+1
    K_h = (np.float64(2.0)**(np.linspace(L_min-1, L_max-1, L).astype(np.float64)))*np.float64(K_star)
    return L, K_h

def init_lds(X_hds, N, init='pca', n_components=2, rand_state=None):
    """
    Initialize the LD embedding.
    In:
    - X_hds: numpy.ndarray with shape (N, M), containing the HD data set, with one example per row and one dimension per column, or None. If X_hds is set to None, init cannot be equal to 'pca', otherwise an error is raised. 
    - N: number of examples in the data set. If X_hds is not None, N must be equal to X_hds.shape[0]. 
    - init: determines the initialization of the LD embedding. 
    ---> If isinstance(init, str) is True:
    ------> If init is equal to 'pca', the LD embedding is initialized with the first n_components principal components of X_hds. X_hds cannot be None in this case, otherwise an error is raised. 
    ------> If init is equal to 'random', the LD embedding is initialized randomly, using a uniform Gaussian distribution with small variance. X_hds may be set to None in this case.
    ------> Otherwise an error is raised.
    ---> If isinstance(init, np.ndarray) is True:
    ------> init must in this case be a 2-D numpy array, with N rows and n_components columns. It stores the LD positions to use for the initialization, with one example per row and one LD dimension per column. init[i,:] contains the initial LD coordinates for the HD sample X_hds[i,:]. X_hds may be set to None in this case. If init.ndim != 2 or init.shape[0] != N or init.shape[1] != n_components, an error is raised.
    ---> Otherwise, an error is raised.
    - n_components: number of dimensions in the LD space.
    - rand_state: random state. If it is None, it is set to np.random.
    Out:
    A numpy ndarray with shape (N, n_components), containing the initialization of the LD data set, with one example per row and one LD dimension per column.
    """
    global module_name
    if rand_state is None:
        rand_state = np.random
    if isinstance(init, str):
        if init == "pca":
            if X_hds is None:
                raise ValueError("Error in function init_lds of module {module_name}: init cannot be set to 'pca' if X_hds is None.".format(module_name=module_name))
            return sklearn.decomposition.PCA(n_components=n_components, whiten=False, copy=True, svd_solver='auto', iterated_power='auto', tol=0.0, random_state=rand_state).fit_transform(X_hds)
        elif init == 'random':
            return 10.0**(-4) * rand_state.randn(N, n_components)
        else:
            raise ValueError("Error in function init_lds of module {module_name}: unknown value '{init}' for init parameter.".format(module_name=module_name, init=init))
    elif isinstance(init, np.ndarray):
        if init.ndim != 2:
            raise ValueError("Error in function init_lds of module {module_name}: init must be 2-D.".format(module_name=module_name))
        if init.shape[0] != N:
            raise ValueError("Error in function init_lds of module {module_name}: init must have {N} rows, but init.shape[0] = {v}.".format(module_name=module_name, N=N, v=init.shape[0]))
        if init.shape[1] != n_components:
            raise ValueError("Error in function init_lds of module {module_name}: init must have {n_components} columns, but init.shape[1] = {v}.".format(module_name=module_name, n_components=n_components, v=init.shape[1]))
        return init
    else:
        raise ValueError("Error in function init_lds of module {module_name}: unknown type value '{v}' for init parameter.".format(module_name=module_name, v=type(init)))

@numba.jit(nopython=True)
def ms_ld_bandwidths(tau_hi, K_h, N, n_components, X_lds, fit_U=True):
    """
    Compute the multi-scale LD bandwidths and precisions.
    In:
    - tau_hi: a 2-D numpy array in which element tau_hi[h,i] contains tau_{hi} = 2/pi_{hi}, as defined in [2]. 
    - K_h: 1-D numpy array, with the perplexities at each scale in increasing order.
    - N: number of data points.
    - n_components: number of components of the LDS.
    - X_lds: 2-D numpy array with shape (N, n_components) containing the current value of the LD embedding.
    - fit_U: boolean. If True, U is computed as defined in [2]. Otherwise, it is fixed to 1 and tau_hi is not used and its value does not matter.
    Out:
    A tuple with:
    - D_h: if fit_U is True, a 1-D numpy array containing D_{h} at the different scales as defined in [2]. Otherwise, equal to np.empty(shape=tau_hi.shape[0]-1, dtype=np.float64).
    - U: if fit_U is True, U as defined in [2]. Otherwise, equal to 1.0.
    - p_h: 1-D numpy array with the LD precisions at each scale defined by K_h.
    - t_h: 1-D numpy array with the LD bandwidths at each scale defined by K_h.
    """
    global n_eps_np_float64
    N_f = np.float64(N)
    n_c_f = np.float64(n_components)
    # Fitting U as defined in [2].
    if fit_U:
        log_tau_diff = np.log2(tau_hi[1:,:])-np.log2(tau_hi[:-1,:])
        log_tau_diff_neq0 = np.nonzero(log_tau_diff!=0)[0]
        log_tau_diff[log_tau_diff_neq0] = 1.0/log_tau_diff[log_tau_diff_neq0]
        D_h = np.dot(log_tau_diff, np.ones(shape=N, dtype=np.float64))*2.0/N_f
        U = np.float64(min(2, max(1, D_h.max()/n_c_f)))
    else:
        D_h = np.empty(shape=K_h.size-1, dtype=np.float64)
        U = 1.0
    # Computing the mean variance of the LD dimensions.
    mean_var_X_lds = np.float64(0.0)
    N_1_f = N_f/np.float64(N-1)
    for k in range(n_components):
        mean_var_X_lds += np.var(X_lds[:,k])*N_1_f
    mean_var_X_lds /= n_c_f
    # Computing the LD precisions
    p_h = K_h**(U*2.0/n_c_f)
    p_h = ((2.0**(1.0+2.0/n_c_f))*p_h.max()/np.maximum(n_eps_np_float64, p_h*mean_var_X_lds)).astype(np.float64)
    # Computing the LD bandwidths. We take the maximum with n_eps_np_float64 as t_h will divide some other quantities.
    t_h = np.maximum(n_eps_np_float64, 2.0/np.maximum(n_eps_np_float64, p_h))
    return D_h, U, p_h, t_h

@numba.jit(nopython=True)
def sne_sim(dsi, vi, i, compute_log=True):
    """
    Compute the SNE asymmetric similarities, as well as their log.
    N refers to the number of data points. 
    In:
    - dsi: numpy 1-D array of floats with N squared distances with respect to data point i. Element k is the squared distance between data points k and i.
    - vi: bandwidth of the exponentials in the similarities with respect to i.
    - i: index of the data point with respect to which the similarities are computed, between 0 and N-1.
    - compute_log: boolean. If True, the logarithms of the similarities are also computed, and otherwise not.
    Out:
    A tuple with two elements:
    - A 1-D numpy array of floats with N elements. Element k is the SNE similarity between data points i and k.
    - If compute_log is True, a 1-D numpy array of floats with N element. Element k is the log of the SNE similarity between data points i and k. By convention, element i is set to 0. If compute_log is False, it is set to np.empty(shape=N, dtype=np.float64).
    """
    N = dsi.size
    si = np.empty(shape=N, dtype=np.float64)
    si[i] = 0.0
    log_si = np.empty(shape=N, dtype=np.float64)
    indj = arange_except_i(N=N, i=i)
    dsij = dsi[indj]
    log_num_sij = (dsij.min()-dsij)/vi
    si[indj] = np.exp(log_num_sij)
    den_si = si.sum()
    si /= den_si
    if compute_log:
        log_si[i] = 0.0
        log_si[indj] = log_num_sij - np.log(den_si)
    return si, log_si

@numba.jit(nopython=True)
def sne_bsf(dsi, vi, i, log_perp):
    """
    Function on which a binary search is performed to find the HD bandwidth of the i^th data point in SNE.
    In: 
    - dsi, vi, i: same as in sne_sim function.
    - log_perp: logarithm of the targeted perplexity.
    Out:
    A float corresponding to the current value of the entropy of the similarities with respect to i, minus log_perp.
    """
    si, log_si = sne_sim(dsi=dsi, vi=vi, i=i, compute_log=True)
    return -np.dot(si, log_si) - log_perp

@numba.jit(nopython=True)
def sne_bs(dsi, i, log_perp, x0=1.0):
    """
    Binary search to find the root of sne_bsf over vi. 
    In:
    - dsi, i, log_perp: same as in sne_bsf function.
    - x0: starting point for the binary search. Must be strictly positive.
    Out:
    A strictly positive float vi such that sne_bsf(dsi, vi, i, log_perp) is close to zero. 
    """
    fx0 = sne_bsf(dsi=dsi, vi=x0, i=i, log_perp=log_perp)
    if close_to_zero(v=fx0):
        return x0
    elif not np.isfinite(fx0):
        raise ValueError("Error in function sne_bs: fx0 is nan.")
    elif fx0 > 0:
        x_up, x_low = x0, x0/2.0
        fx_low = sne_bsf(dsi=dsi, vi=x_low, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_low):
            return x_low
        elif not np.isfinite(fx_low):
            # WARNING: cannot find a valid root!
            return x_up
        while fx_low > 0:
            x_up, x_low = x_low, x_low/2.0
            fx_low = sne_bsf(dsi=dsi, vi=x_low, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_low):
                return x_low
            if not np.isfinite(fx_low):
                return x_up
    else: 
        x_up, x_low = x0*2.0, x0
        fx_up = sne_bsf(dsi=dsi, vi=x_up, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_up):
            return x_up
        elif not np.isfinite(fx_up):
            return x_low
        while fx_up < 0:
            x_up, x_low = 2.0*x_up, x_up
            fx_up = sne_bsf(dsi=dsi, vi=x_up, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_up):
                return x_up
    while True:
        x = (x_up+x_low)/2.0
        fx = sne_bsf(dsi=dsi, vi=x, i=i, log_perp=log_perp)
        if close_to_zero(v=fx):
            return x
        elif fx > 0:
            x_up = x
        else:
            x_low = x

@numba.jit(nopython=True)
def sne_hd_similarities(dsm_hds, perp, compute_log=True, start_bs=np.ones(shape=1, dtype=np.float64)):
    """
    Computes the matrix of SNE asymmetric HD similarities, as well as their log.
    In:
    - dsm_hds: 2-D numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HD distance between i and j.
    - perp: perplexity. Must be > 1.
    - compute_log: boolean. If true, the logarithms of the similarities are also computed. Otherwise not.
    - start_bs: 1-D numpy array with N elements. Element at index i is the starting point of the binary search for the ith data point. If start_bs has only one element, it will be set to np.ones(shape=N, dtype=np.float64).
    Out:
    A tuple with three elements:
    - A 2-D numpy array with shape (N, N) and in which element [i,j] = the HD similarity between i and j. The similarity between i and i is set to 0.
    - If compute_log is True, 2-D numpy array with shape (N, N) and in which element [i,j] = the log of the HD similarity between i and j. By convention, log(0) is set to 0. If compute_log is False, it is set to np.empty(shape=(N,N), dtype=np.float64).
    - A 1-D numpy array with N elements, where element i is the denominator of the exponentials of the HD similarities with respect to data point i. 
    """
    if perp <= 1:
        raise ValueError("""Error in function sne_hd_similarities of module dr_missing_data.py: the perplexity should be >1.""")
    N = dsm_hds.shape[0]
    if start_bs.size == 1:
        start_bs = np.ones(shape=N, dtype=np.float64)
    log_perp = np.log(min(np.float64(perp), np.floor(0.99*np.float64(N))))
    # Computing the N**2 HD similarities, for i, j = 0, ..., N-1.
    si = np.empty(shape=(N,N), dtype=np.float64)
    # Even when compute_log is False, we can not set log_si to None. We need to define it as an array, to be compatible with numba.
    log_si = np.empty(shape=(N,N), dtype=np.float64)
    arr_vi = np.empty(shape=N, dtype=np.float64)
    for i in range(N):
        # Computing the denominator of the exponentials of the HD similarities with respect to data point i. 
        vi = sne_bs(dsi=dsm_hds[i,:], i=i, log_perp=log_perp, x0=start_bs[i])
        # Computing the HD similarities between i and j for j=0, ..., N-1.
        tmp = sne_sim(dsi=dsm_hds[i,:], vi=vi, i=i, compute_log=compute_log)
        si[i,:] = tmp[0]
        if compute_log:
            log_si[i,:] = tmp[1]
        arr_vi[i] = vi
    return si, log_si, arr_vi

@numba.jit(nopython=True)
def ms_hd_similarities(dsm_hds, arr_perp):
    """
    Compute the matrix of multi-scale HD similarities sigma_{ij}, as defined in [2].
    In:
    - dsm_hds: 2-D numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HD distance between i and j.
    - arr_perp: numpy 1-D array containing the perplexities for all scales. All the perplexities must be > 1.
    Out:
    A tuple with:
    - A 2-D numpy array with shape (N, N) and in which element [i,j] = the multi-scale HD similarity sigma_{ij}.
    - A 2-D numpy array with shape (arr_perp.size, N) and in which element [h,i] = tau_{hi} = 2/pi_{hi}, following the notations of [2].
    - sim_hij: 3-D numpy array with shape (arr_perp.size, N, N) where sim_hij[h,:,:] contains the HD similarities at scale arr_perp[h].
    """
    # Number of data points
    N = dsm_hds.shape[0]
    # Number of perplexities
    L = arr_perp.size
    # Matrix storing the multi-scale HD similarities sigma_{ij}. Element [i,j] contains sigma_{ij}. sigma_{ii} is set to 0.
    sigma_ij = np.zeros(shape=(N,N), dtype=np.float64)
    # Matrix storing the HD similarities sigma_{hij} at each scale.
    sim_hij = np.empty(shape=(L,N,N), dtype=np.float64)
    # Matrix storing the HD tau_{hi}. Element [h,i] contains tau_{hi}.
    tau_hi = np.empty(shape=(L,N), dtype=np.float64)
    # For each perplexity
    for h, perp in enumerate(arr_perp):
        # Using the bandwidths found at the previous scale to initialize the binary search at the current scale.
        if h > 0:
            start_bs = tau_hi[h-1,:]
        else:
            start_bs = np.ones(shape=N, dtype=np.float64)
        # Computing the N**2 HD similarities sigma_{hij}
        sim_hij[h,:,:], dum, tau_hi[h,:] = sne_hd_similarities(dsm_hds=dsm_hds, perp=perp, compute_log=False, start_bs=start_bs)
        # Updating the multi-scale HD similarities
        sigma_ij += sim_hij[h,:,:]
    # Scaling the multi-scale HD similarities
    sigma_ij /= np.float64(L)
    # Returning
    return sigma_ij, tau_hi, sim_hij

@numba.jit(nopython=True)
def ms_hld_bandwidths(dsm_hds, K_h, N, n_components, X_lds, fit_U=True):
    """
    Compute the multi-scale HD and LD bandwidths and precisions.
    In:
    - dsm_hds: 2-D numpy array with the squared pairwise HD distances between the data points.
    - K_h: 1-D numpy array, with the perplexities at each scale in increasing order.
    - N: number of data points.
    - n_components: number of components of the LDS.
    - X_lds: 2-D numpy array with shape (N, n_components) containing the current value of the LD embedding.
    - fit_U: same as for ms_ld_bandwidths.
    Out:
    A tuple with:
    - tau_hi: a 2-D numpy array in which element tau_hi[h,i] contains tau_{hi} = 2/pi_{hi}, as defined in [2].
    - D_h: if fit_U is True, a 1-D numpy array containing D_{h} at the different scales as defined in [2]. Otherwise, a dummy 1-D numpy array of the same size as when fit_U is True.
    - U: U as defined in [2] if fit_U is True, or 1 otherwise.
    - p_h: 1-D numpy array with the LD precisions at each scale defined by K_h.
    - t_h: 1-D numpy array with the LD bandwidths at each scale defined by K_h.
    """
    # Computing the multi-scale HD similarities. Element sigma_ij[i,j] contains sigma_{ij}. sigma_{ii} is set to 0. Element tau_hi[h,i] contains tau_{hi} = 2/pi_{hi}. 
    sigma_ij, tau_hi = ms_hd_similarities(dsm_hds=dsm_hds, arr_perp=K_h)[:2]
    # Computing the LD bandwidths and precisions
    D_h, U, p_h, t_h = ms_ld_bandwidths(tau_hi=tau_hi, K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U)
    # Returning
    return tau_hi, D_h, U, p_h, t_h

@numba.jit(nopython=True)
def mssne_lds_similarities_h(arr_den_s_i, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t):
    """
    Computation of the matrix of Ms SNE asymmetric similarities in LDS at some scale, as well as their log.
    We denote by
    - dsm_lds: 2-D numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared LD distance between i and j. The diagonal elements are assumed to be equal to np.inf.
    - dsm_lds_min_row: equal to dsm_lds.min(axis=1).
    - N: number of data points.
    In:
    - arr_den_s_i: numpy 1-D array with N elements, where element i is the denominator of the exponentials of the LD similarities between i and j for j=0, ..., N-1. It is assumed that arr_den_s_i >= np_eps.
    - np_eps: equal to np.finfo(dtype=np.float64).eps.
    - arr_ones: equal to np.ones(shape=N, dtype=np.float64).
    - dsm_lds_min_row_dsm_lds_t: equal to dsm_lds_min_row-dsm_lds.T.
    Out:
    A 2-D numpy array with shape (N, N) and in which element [i,j] = the LD similarity between i and j. The LD similarity between i and i is set to 0.
    """
    # Numerators of the similarities
    s_i = np.exp(dsm_lds_min_row_dsm_lds_t/arr_den_s_i)
    # Correcting the diagonal of the similarities.
    s_i = fill_diago(s_i, 0.0).astype(np.float64)
    # Computing the N**2 LD similarities, for i, j = 0, ..., N-1, and returning.
    return (s_i/np.maximum(np_eps, np.dot(arr_ones.astype(np.float64), s_i))).T

@numba.jit(nopython=True)
def mssne_sev_lds_similarities_h(arr_den_s_i, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t):
    """
    Similar to mssne_lds_similarities_h, but for several data sets.
    In:
    - np_eps, arr_ones: same as for mssne_lds_similarities_h.
    - arr_den_s_i: 2-D numpy array with shape (n_samp, N). For each k in range(n_samp), arr_den_s_i[k,:] must be a valid arr_den_s_i argument for mssne_lds_similarities_h.
    - dsm_lds_min_row_dsm_lds_t: 3-D numpy array with shape (n_samp, N, N). For each k in range(n_samp), dsm_lds_min_row_dsm_lds_t[k,:,:] must be a valid dsm_lds_min_row_dsm_lds_t argument for mssne_lds_similarities_h.
    Out:
    A 3-D numpy array ret_v with shape (n_samp, N, N). For each k in range(n_samp), ret_v[k,:,:] is equal to mssne_lds_similarities_h(arr_den_s_i=arr_den_s_i[k,:], np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_lds_min_row_dsm_lds_t[k,:,:]).
    """
    # Number of data sets and number of examples per data set
    n_samp, N = dsm_lds_min_row_dsm_lds_t.shape[:2]
    # Initializing the array to return
    ret_v = np.empty(shape=(n_samp, N, N), dtype=np.float64)
    # For each data set
    for k in range(n_samp):
        ret_v[k,:,:] = mssne_lds_similarities_h(arr_den_s_i=arr_den_s_i[k,:], np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_lds_min_row_dsm_lds_t[k,:,:])
    # Returning
    return ret_v

@numba.jit(nopython=True)
def mssne_eval_sim_lds(N, n_perp, t_h, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t):
    """
    Evaluates the LD similarities.
    In: 
    - N: number of data points.
    - n_perp: number of perplexities which are considered.
    - t_h: 1-D numpy array containing n_perp elements and in which element h contains the LD bandwidth associated with the h^th considered perplexity, which is equal to 2.0/p_h[h]. It is assumed that t_h >= np_eps.
    - np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t: as in function mssne_lds_similarities_h.
    Out:
    A tuple with:
    - A 2-D numpy array with shape (N, N) containing the pairwise multi-scale LD similarities. The diagonal elements are forced to 1.0.
    - A 3-D numpy array s_hij with shape (n_perp, N, N). For each h in range(n_perp), s_hij[h,:,:] contains the pairwise LD similarities at scale h.
    """
    # LD single-scale similarities.
    s_hij = np.empty(shape=(n_perp,N,N), dtype=np.float64)
    # LD multi-scale similarities.
    s_ij = np.zeros(shape=(N,N), dtype=np.float64)
    # For each scale
    for h in range(n_perp):
        # Computing the corresponding LD similarities and updating s_ij
        s_hij[h,:,:] = mssne_lds_similarities_h(arr_den_s_i=t_h[h]*arr_ones, np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_lds_min_row_dsm_lds_t)
        s_ij += s_hij[h,:,:]
    # Scaling s_ij
    s_ij /= np.float64(n_perp)
    # Since s_ij is only used as a denominator, we fill its diagonal with ones, to avoid dividing by zero. This does not change the results, as the diagonal of sigma_ij is equal to 0.
    s_ij = fill_diago(s_ij, 1.0)
    # Setting the remaining 0 elements of s_ij to the smallest non-zero value, to avoid dividing by zero.
    s_ij = np.maximum(np_eps, s_ij)
    # Returning
    return s_ij, s_hij

@numba.jit(nopython=True)
def mssne_sev_eval_sim_lds(N, n_perp, t_h, n_samp, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t):
    """
    Evaluate mssne_eval_sim_lds for several data sets.
    In:
    - N, n_perp, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t: same as for mssne_eval_sim_lds.
    - t_h: 2-D numpy array with shape (n_samp, n_perp). For each k in range(n_samp), t_h[k,:] contains the LD bandwidths at each scale. It is assumed that t_h >= np_eps.
    - n_samp: strictly positive integer indicating the number of data sets for which the LD similarities must be computed.
    Out:
    A 3-D numpy array s_ij with shape (n_samp, N, N). For each i in range(n_samp), s_ij[i,:,:] contains the result of mssne_eval_sim_lds(N=N, n_perp=n_perp, t_h=t_h[i,:], np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_lds_min_row_dsm_lds_t).
    """
    # Initializing the similarities to return
    s_ij = np.empty(shape=(n_samp, N, N), dtype=np.float64)
    # For each data set
    for i in range(n_samp):
        s_ij[i,:,:] = mssne_eval_sim_lds(N=N, n_perp=n_perp, t_h=t_h[i,:], np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_lds_min_row_dsm_lds_t)[0]
    # Returning
    return s_ij

def mssne_eval_dsm_lds_min_row_dsm_lds_t_dsm(dsm_lds):
    """
    Evaluates the dsm_lds_min_row_dsm_lds_t parameter of function mssne_lds_similarities_h. N denotes the number of samples.
    In: 
    - dsm_lds: numpy 2-D array with shape (N, N), containing the pairwise LD squared distances.
    Out: 
    dsm_lds_min_row_dsm_lds_t as described in mssne_lds_similarities_h.
    """
    np.fill_diagonal(a=dsm_lds, val=np.inf)
    # Returning
    return dsm_lds.min(axis=1)-dsm_lds.T

def mssne_sev_eval_dsm_lds_min_row_dsm_lds_t_dsm(dsm_lds):
    """
    Similar to mssne_eval_dsm_lds_min_row_dsm_lds_t_dsm but for several distance matrices.
    In:
    - dsm_lds: 3-D numpy array with shape (n_samp, N, N). For each k in range(n_samp), dsm_lds[k,:,:] must be a valid argument for function mssne_eval_dsm_lds_min_row_dsm_lds_t_dsm.
    Out:
    A 3-D numpy array ret_v with shape (n_samp, N, N). For each k in range(n_samp), ret_v[k,:,:] is equal to mssne_eval_dsm_lds_min_row_dsm_lds_t_dsm(dsm_lds=dsm_lds[k,:,:]).
    """
    # Number of data sets and number of examples per data set
    n_samp, N = dsm_lds.shape[:2]
    # Initializing the array to return
    ret_v = np.empty(shape=(n_samp, N, N), dtype=np.float64)
    # For each data set
    for k in range(n_samp):
        ret_v[k,:,:] = mssne_eval_dsm_lds_min_row_dsm_lds_t_dsm(dsm_lds=dsm_lds[k,:,:])
    # Returning
    return ret_v

def mssne_eval_dsm_lds_min_row_dsm_lds_t(X):
    """
    Evaluate the dsm_lds_min_row_dsm_lds_t parameter of function mssne_lds_similarities_h.
    In: 
    - X: numpy 2-D array with shape (N, n_components), containing the current values of the LD coordinates. It contains one example per row and one LD dimension per column.
    Out: 
    dsm_lds_min_row_dsm_lds_t as described in mssne_lds_similarities_h.
    """
    # Computing the pairwise squared Euclidean distances in the LDS
    dsm_lds = scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X, metric='sqeuclidean'), force='tomatrix')
    # Returning
    return mssne_eval_dsm_lds_min_row_dsm_lds_t_dsm(dsm_lds=dsm_lds)

def mssne_obj_fct(x, sigma_ij, N, n_components, p_h, t_h, n_perp):
    """
    Compute the value of the objective function of Multi-scale SNE.
    In:
    - x: numpy 1-D array with N*n_components elements, containing the current values of the LD coordinates. np.reshape(a=x, newshape=(N, n_components)) should yield a 2-D array with one example per row and one LD dimension per column.
    - sigma_ij: numpy 2-D array with shape (N,N). Element (i,j) should contain sigma_{ij}, as defined in [2]. Diagonal elements must be equal to 0.
    - N: number of data points.
    - n_components: dimension of the LDS.
    - n_perp: number of perplexities which are considered.
    - p_h: 1-D numpy array containing n_perp elements and in which element h contains the LD precision associated with the h^th considered perplexity.
    - t_h: 1-D numpy array containing n_perp elements and in which element h contains the LD bandwidth associated with the h^th considered perplexity, which is equal to 2.0/p_h[h]. It is assumed that t_h >= np_eps.
    Out:
    A scalar equal to the Multi-scale SNE objective function value.
    Remark:
    - To use scipy optimization functions, the functions mssne_obj_fct and mssne_grad should have the same arguments.
    """
    global n_eps_np_float64
    # Evaluating the LD similarities.
    s_ij = mssne_eval_sim_lds(N=N, n_perp=n_perp, t_h=t_h, np_eps=n_eps_np_float64, arr_ones=np.ones(shape=N, dtype=np.float64), dsm_lds_min_row_dsm_lds_t=mssne_eval_dsm_lds_min_row_dsm_lds_t(X=np.reshape(a=x, newshape=(N, n_components))))[0]
    # Computing the cost function value
    return scipy.special.rel_entr(sigma_ij, s_ij).sum()

def mssne_sev_obj_fct(x, sigma_ij, N, n_components, p_h, t_h, n_perp):
    """
    Evaluate the objective function of Ms SNE for multiple HD similarities and return the mean of the evaluations.
    In: 
    - sigma_ij: numpy 3-D array with shape (n_samp, N, N). For each k in range(n_samp), sigma_ij[k,i,j] should contain sigma_{ij}, as defined in [2], for the k^th data set. Elements for which i=j must be equal to 0.
    - p_h: 2-D numpy array with shape (n_samp, n_perp). For each k in range(n_samp), p_h[k,:] contains the LD precisions at each scale.
    - t_h: 2-D numpy array with shape (n_samp, n_perp). For each k in range(n_samp), t_h[k,:] contains the LD bandwidths at each scale. It is assumed that t_h >= np_eps.
    - x, N, n_components, n_perp: same as in mssne_obj_fct.
    Out:
    A float being the mean of the calls mssne_obj_fct(x=x, sigma_ij=sigma_ij[k,:,:], N=N, n_components=n_components, p_h=p_h[k,:], t_h=t_h[k,:], n_perp=n_perp) for each k in range(n_samp).
    """
    global n_eps_np_float64
    n_samp = sigma_ij.shape[0]
    # Evaluating the LD similarities for the different data sets. For each i in range(sigma_ij.shape[0]), s_ij[i,:,:] has its diagonal elements forced to zero.
    s_ij = mssne_sev_eval_sim_lds(N=N, n_perp=n_perp, t_h=t_h, n_samp=n_samp, np_eps=n_eps_np_float64, arr_ones=np.ones(shape=N, dtype=np.float64), dsm_lds_min_row_dsm_lds_t=mssne_eval_dsm_lds_min_row_dsm_lds_t(X=np.reshape(a=x, newshape=(N, n_components))))
    # Computing the mean cost function value
    return np.sum(scipy.special.rel_entr(sigma_ij, s_ij))/np.float64(n_samp)

@numba.jit(nopython=True)
def mssne_eval_grad(N, n_perp, t_h, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t, n_components, p_h, X, sigma_ij):
    """
    Evaluate the Ms SNE gradient.
    In:
    - N: number of data points.
    - n_perp: number of perplexities which are considered.
    - t_h: 1-D numpy array containing n_perp elements and in which element h contains the LD bandwidth associated with the h^th considered perplexity, which is equal to 2.0/p_h[h]. It is assumed that t_h >= np_eps.
    - np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t: same as in function mssne_lds_similarities_h.
    - n_components: dimension of the LDS.
    - p_h: 1-D numpy array containing n_perp elements and in which element h contains the LD precision associated with the h^th considered perplexity.
    - X: numpy 2-D array with shape (N, n_components), containing the current values of the LD coordinates. It contains one example per row and one LD dimension per column.
    - sigma_ij: numpy 3-D array with shape (n_samp, N, N). For each k in range(n_samp), sigma_ij[k,i,j] should contain sigma_{ij}, as defined in [2], for the k^th data set. Elements for which i=j must be equal to 0.
    Out:
    A 2-D numpy array with shape (N, n_components) containing the evaluation of the Ms SNE gradient.
    """
    # Computing the LD similarities
    s_ij, s_hij = mssne_eval_sim_lds(N=N, n_perp=n_perp, t_h=t_h, np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_lds_min_row_dsm_lds_t)
    # Computing the quotient of sigma_ij by s_ij
    ss_ij = sigma_ij/s_ij
    # Computing the gradient
    grad = np.zeros(shape=(N, n_components), dtype=np.float64)
    for h in range(n_perp):
        # Computing the product between ss_ij and s_hij[h,:,:], summing over the columns, substracting the result from each row of ss_ij and multiplying by s_hij[h,:,:].
        Mh = s_hij[h,:,:]*((ss_ij.T - np.dot(ss_ij*s_hij[h,:,:], arr_ones)).T)
        Mh += Mh.T
        # Updating the gradient
        grad += p_h[h]*((X.T*np.dot(Mh, arr_ones)).T - np.dot(Mh, X))
    # Returning
    return grad/np.float64(n_perp)

@numba.jit(nopython=True)
def mssne_eval_sev_grad(N, n_perp, t_h, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t, n_components, p_h, X, sigma_ij, n_samp):
    """
    Evaluate the mean Ms SNE gradient over several data sets.
    In: 
    - N, n_perp, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t, n_components, X: same as for mssne_eval_grad.
    - t_h: 2-D numpy array with shape (n_samp, n_perp). For each k in range(n_samp), t_h[k,:] contains the LD bandwidths at each scale. It is assumed that t_h >= np_eps.
    - n_samp: strictly positive integer indicating the number of data sets for which the similarities must be computed.
    - sigma_ij: numpy 3-D array with shape (n_samp, N, N). For each k in range(n_samp), sigma_ij[k,i,j] should contain sigma_{ij}, as defined in [2], for the k^th data set. Elements for which i=j must be equal to 0.
    - p_h: 2-D numpy array with shape (n_samp, n_perp). For each k in range(n_samp), p_h[k,:] contains the LD precisions at each scale.
    Out:
    A 2-D numpy array with shape (N, n_components) containing the mean of the Ms SNE gradients of the n_samp data sets provided through the arguments.
    """
    # Initializing the gradient
    grad = np.zeros(shape=(N, n_components), dtype=np.float64)
    # For each data set
    for i in range(n_samp):
        grad += mssne_eval_grad(N=N, n_perp=n_perp, t_h=t_h[i,:], np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_lds_min_row_dsm_lds_t, n_components=n_components, p_h=p_h[i,:], X=X, sigma_ij=sigma_ij[i,:,:])
    # Returning the gradient mean
    return grad/np.float64(n_samp)

def mssne_grad(x, sigma_ij, N, n_components, p_h, t_h, n_perp):
    """
    Compute the value of the gradient of the objective function of Multi-scale SNE.
    In:
    - x: numpy 1-D array with N*n_components elements, containing the current values of the LD coordinates. np.reshape(a=x, newshape=(N, n_components)) should yield a 2-D array with one example per row and one LD dimension per column.
    - sigma_ij: numpy 2-D array with shape (N,N). Element (i,j) should contain sigma_{ij}, as defined in [2]. Diagonal elements must be equal to 0.
    - N: number of data points.
    - n_components: dimension of the LDS.
    - n_perp: number of perplexities which are considered.
    - p_h: 1-D numpy array containing n_perp elements and in which element h contains the LD precision associated with the h^th considered perplexity.
    - t_h: 1-D numpy array containing n_perp elements and in which element h contains the LD bandwidth associated with the h^th considered perplexity, which is equal to 2.0/p_h[h]. It is assumed that t_h >= n_eps_np_float64.
    Out:
    A 1-D numpy array with N*n_components elements, where element i is the coordinate of the gradient associated to x[i].
    Remark:
    - In order to use the scipy optimization functions, the functions mssne_obj_fct and mssne_grad should have the same arguments.
    """
    global n_eps_np_float64
    X = np.reshape(a=x, newshape=(N, n_components))
    # Evaluating the gradient
    grad = mssne_eval_grad(N=N, n_perp=n_perp, t_h=t_h, np_eps=n_eps_np_float64, arr_ones=np.ones(shape=N, dtype=np.float64), dsm_lds_min_row_dsm_lds_t=mssne_eval_dsm_lds_min_row_dsm_lds_t(X=X), n_components=n_components, p_h=p_h, X=X, sigma_ij=sigma_ij)
    # Returning the reshaped gradient
    return np.reshape(a=grad, newshape=N*n_components)

def mssne_sev_grad(x, sigma_ij, N, n_components, p_h, t_h, n_perp):
    """
    Evaluate the gradient of Ms SNE for multiple HD similarities and return the mean of the evaluations.
    In: 
    - sigma_ij: numpy 3-D array with shape (n_samp, N, N). For each k in range(n_samp), sigma_ij[k,i,j] should contain sigma_{ij}, as defined in [2], for the k^th data set. Elements for which i=j must be equal to 0.
    - p_h: 2-D numpy array with shape (n_samp, n_perp). For each k in range(n_samp), p_h[k,:] contains the LD precisions at each scale.
    - t_h: 2-D numpy array with shape (n_samp, n_perp). For each k in range(n_samp), t_h[k,:] contains the LD bandwidths at each scale. It is assumed that t_h >= n_eps_np_float64.
    - x, N, n_components, n_perp: same as in mssne_grad.
    Out:
    A 1-D numpy array with N*n_components elements, where element i is the mean coordinate of the gradient associated to x[i].
    """
    X = np.reshape(a=x, newshape=(N, n_components))
    # Evaluating the gradient
    grad = mssne_eval_sev_grad(N=N, n_perp=n_perp, t_h=t_h, np_eps=np.finfo(dtype=np.float64).eps, arr_ones=np.ones(shape=N, dtype=np.float64), dsm_lds_min_row_dsm_lds_t=mssne_eval_dsm_lds_min_row_dsm_lds_t(X=X), n_components=n_components, p_h=p_h, X=X, sigma_ij=sigma_ij, n_samp=sigma_ij.shape[0])
    # Returning the reshaped gradient
    return np.reshape(a=grad, newshape=N*n_components)

def mssne_sim_hds_bandwidth(X_hds, K_h, N, n_components, X_lds, fit_U=True, dsm_hds=None, dm_fct=None):
    """
    Evaluate the multi-scale HD and LD bandwidths and precisions, or directly return the HD similarities at the different scales.
    In:
    - X_hds: 2-D numpy.ndarray with shape (N, M), containing the HD data set, with one example per row and one dimension per column. If None, dsm_hds must be specified and different from None, otherwise an error is raised. X_hds is only used if dsm_hds is set to None.
    - K_h: 1-D numpy array with the perplexities at each scale in increasing order.
    - N: number of data points.
    - n_components: number of components in the LDS.
    - X_lds: 2-D numpy array with shape (N, n_components) containing the current value of the LD embedding.
    - fit_U: same as for ms_ld_bandwidths.
    - dsm_hds: (optional) 2-D numpy.ndarray with shape (N,N) containing the pairwise SQUARED HD distances between the data points in X_hds. dsm_hds[i,j] stores the SQUARED HD distance between X_hds[i,:] and X_hds[j,:]. If dsm_hds is specified and not None, X_hds is not used and can be set to None. If dsm_hds is None, it is deduced from X_hds using squared Euclidean distances if dm_fct is None, or using dm_fct(X_hds)**2 if dm_fct is not None. Hence, if dsm_hds is None, X_hds cannot be None, otherwise an error is raised.
    - dm_fct: (optional) a function taking X_hds as argument and returning a 2-D np.ndarray dm_hds with shape (N,N) containing the pairwise HD distances (NOT squared) between the data points in X_hds. dm_hds[i,j] stores the HD distance (NOT squared) between X_hds[i,:] and X_hds[j,:]. If dsm_hds is specified and not None, X_hds and dm_fct are not used and can be set to None. If dsm_hds is None, it is deduced from X_hds using squared Euclidean distance if dm_fct is None, or using dm_fct(X_hds)**2 if dm_fct is not None. Hence, dm_fct is only used if dsm_hds is None and in this case, X_hds cannot be None, otherwise an error is raised. An example of a valid function for the dm_fct parameter is the eucl_dist_matr one.
    Out:
    If fit_U is True, a tuple with:
    - dsm_hds: 2-D numpy array with the squared pairwise HD distances between the data points.
    - tau_hi: a 2-D numpy array in which element tau_hi[h,i] contains tau_{hi} = 2/pi_{hi}, as defined in [2].
    - p_h: 1-D numpy array with the LD precisions at each scale defined by K_h.
    - t_h: 1-D numpy array with the LD bandwidths at each scale defined by K_h.
    If fit_U is False, return a 3-D numpy array sim_hij with shape (K_h.size, N, N) where sim_hij[h,:,:] contains the HD similarities at scale K_h[h].
    """
    global module_name
    # Computing the squared HD distances. 
    if dsm_hds is None:
        if X_hds is None:
            raise ValueError("Error in function mssne_sim_hds_bandwidth of module {module_name}: if dsm_hds is None, X_hds cannot be None.".format(module_name=module_name))
        if dm_fct is None:
            dsm_hds = scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X_hds, metric='sqeuclidean'), force='tomatrix')
        else:
            dsm_hds = (dm_fct(X_hds)**2).astype(np.float64)
    if fit_U:
        # Computing the multi-scale HD and LD bandwidths and precisions
        tau_hi, D_h, U, p_h, t_h = ms_hld_bandwidths(dsm_hds=dsm_hds, K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U)
        return dsm_hds, tau_hi, p_h, t_h
    else:
        # Returning the HD similarities at the scales indicated by K_h
        return ms_hd_similarities(dsm_hds=dsm_hds, arr_perp=K_h)[2]

def mssne_sev_sim_hds_bandwidth(X_hds_sev, K_h, N, n_components, X_lds, fit_U=True, dm_fct=None):
    """
    Call the function mssne_sim_hds_bandwidth on multiple HD data sets.
    In:
    - X_hds_sev: 3-D numpy array with shape (n_samp, N, M). For i in range(n_samp), X_hds_sev[i,:,:] contains a HD data set, with one example per row and one dimension per column.
    - K_h: 1-D numpy array with the perplexities at each scale in increasing order. Its size is denoted by L.
    - N: number of data points.
    - n_components: number of components of the LDS.
    - X_lds: 2-D numpy array with shape (N, n_components) containing the current value of the LD embedding.
    - fit_U: same as in mssne_sev_implem.
    - dm_fct: same as in mssne_sim_hds_bandwidth function. See mssne_sim_hds_bandwidth for a description.
    Out:
    If fit_U is True, a tuple with:
    - dsm_hds: 3-D numpy array with shape (n_samp, N, N). For each i in range(n_samp), dsm_hds[i,:,:] contains the squared pairwise HD distances between the data points in X_hds_sev[i,:,:].
    - tau_hi: a 3-D numpy array with shape (n_samp, L, N). For each k in range(n_samp), tau_hi[k,h,i] contains tau_{hi} = 2/pi_{hi}, as defined in [2], for the data set X_hds_sev[k,:,:].
    - p_h: 2-D numpy array with shape (n_samp, L). For each k in range(n_samp), p_h[k,:] contains the LD precisions at each scale defined by K_h for the data set X_hds_sev[k,:,:].
    - t_h: 2-D numpy array with shape (n_samp, L). For each k in range(n_samp), t_h[k,:] contains the LD bandwidths at each scale defined by K_h for the data set X_hds_sev[k,:,:].
    If fit_U is False, a tuple with:
    - sim_hij: a 3-D numpy array with shape (K_h.size, N, N). sim_hij[h,:,:] contains the expectation of the HD similarities at scale K_h[h] over the HD data sets.
    - p_h: 1-D numpy array with K_h.size elements, containing the LD precisions at each scale. 
    - t_h: 1-D numpy array with K_h.size elements, containing the LD bandwidths at each scale. 
    """
    n_samp = X_hds_sev.shape[0]
    # Number of scales
    L = K_h.size
    # Initializing the returned values
    if fit_U:
        dsm_hds, tau_hi, p_h, t_h = np.empty(shape=(n_samp, N, N), dtype=np.float64), np.empty(shape=(n_samp, L, N), dtype=np.float64), np.empty(shape=(n_samp, L), dtype=np.float64), np.empty(shape=(n_samp, L), dtype=np.float64)
    else:
        # Computing the LD bandwidths and precisions. The value of tau_hi does not matter as fit_U is False; it must still be set to a numpy array of np.float64 for numba compatibility.
        p_h, t_h = ms_ld_bandwidths(tau_hi=np.empty(shape=(L, N), dtype=np.float64), K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U)[2:]
        # Initializing the HD similarities at each scale that will be returned
        sim_hij = np.zeros(shape=(L, N, N), dtype=np.float64)
    # For each data set
    for k in range(n_samp):
        ret_k = mssne_sim_hds_bandwidth(X_hds=X_hds_sev[k,:,:], K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U, dm_fct=dm_fct)
        if fit_U:
            dsm_hds[k,:,:], tau_hi[k,:,:], p_h[k,:], t_h[k,:] = ret_k
        else:
            sim_hij += ret_k
    # Returning
    if fit_U:
        return dsm_hds, tau_hi, p_h, t_h
    else:
        # Normalizing sim_hij before returning. This enables computing the expectation of the HD similarities over the HD data sets.
        sim_hij /= np.float64(n_samp)
        return sim_hij, p_h, t_h

def mssne_manage_seed(seed_mssne=None):
    """
    Manage the random seed in mssne_implem and mssne_sev_implem.
    In:
    - seed_mssne: an integer or None. If it is None, it is set to seed_MsSNE_def. If it is not an integer, an error is raised. 
    Out:
    If seed_mssne > 0, np.random.RandomState(seed_mssne) is returned. Otherwise, np.random is returned. 
    """
    global seed_MsSNE_def, module_name
    if seed_mssne is None:
        seed_mssne = seed_MsSNE_def
    if seed_mssne != int(round(seed_mssne)):
        raise ValueError("Error in function mssne_manage_seed of module {module_name}: seed_mssne must be an integer.".format(module_name=module_name))
    if seed_mssne > 0:
        return np.random.RandomState(seed_mssne)
    else:
        return np.random

def mssne_implem(X_hds, init='pca', n_components=2, ret_sim_hds=False, fit_U=True, dm_hds=None, seed_mssne=None):
    """
    This function applies Multi-scale SNE to reduce the dimension of a data set [2].
    In:
    - X_hds: 2-D numpy.ndarray with shape (N, M), containing the HD data set, with one example per row and one dimension per column, or None. If X_hds is not None, it is assumed that it does not contain duplicated examples. X_hds can only be None if dm_hds is not None and init is not set to 'pca', otherwise an error is raised. 
    - init: determines the initialization of the LD embedding. 
    ---> If isinstance(init, str) is True:
    ------> If init is equal to 'pca', the LD embedding is initialized with the first n_components principal components of X_hds. X_hds cannot be None in this case, even if dm_hds is specified, otherwise an error is raised. 
    ------> If init is equal to 'random', the LD embedding is initialized randomly, using a uniform Gaussian distribution with small variance. X_hds may be set to None in this case if dm_hds is specified.
    ------> Otherwise an error is raised.
    ---> If isinstance(init, np.ndarray) is True:
    ------> init must in this case be a 2-D numpy array, with N rows and n_components columns. It stores the LD positions to use for the initialization, with one example per row and one LD dimension per column. init[i,:] contains the initial LD coordinates for the HD sample X_hds[i,:]. X_hds may be set to None in this case if dm_hds is specified. If init.ndim != 2 or init.shape[0] != N or init.shape[1] != n_components, an error is raised.
    ---> Otherwise, an error is raised.
    - n_components: dimension of the LDS.
    - ret_sim_hds: boolean. If True, the multi-scale HD similarities are also returned.
    - fit_U: boolean indicating whether to fit the U in the definition of the LD similarities in [2]. If True, the U is tuned as in [2]. Otherwise, it is forced to 1. Setting fit_U to True usually tends to slightly improve DR quality at the expense of slightly increasing computation time. 
    - dm_hds: (optional) 2-D numpy array with the pairwise HD distances (NOT squared) between the data points. If dm_hds is None, it is deduced from X_hds using Euclidean distances. If dm_hds is not None, then the pairwise HD distances are not recomputed and X_hds may either be None or defined; if X_hds is not None, it will only be used if init is set to 'pca', to initialize the LD embedding to the first n_components principal components of X_hds. Hence, if both dm_hds and X_hds are not None and if init is set to 'pca', dm_hds and X_hds are assumed to be compatible, i.e. dm_hds[i,j] is assumed to store the HD distance between X_hds[i,:] and X_hds[j,:].
    - seed_mssne: seed to use for the random state. Check mssne_manage_seed for a description. 
    Out:
    If ret_sim_hds is False, a 2-D numpy.ndarray X_lds with shape (N, n_components), containing the LD data set, with one example per row and one dimension per column. X_lds[i,:] contains the LD coordinates of the HD sample X_hds[i,:]. 
    If ret_sim_hds is True, a tuple with:
    - X_lds as described in the case of ret_sim_hds = False.
    - a 2-D numpy array sigma_ij with shape (N, N), where N is the number of samples. It contains the multi-scale pairwise HD similarities between the samples. sigma_ij[i,j] stores the multi-scale HD similarity between X_hds[i,:] and X_hds[j,:].
    Remarks:
    - L-BFGS algorithm is used, as suggested in [2].
    - Multi-scale optimization is performed, as presented in [2].
    - Euclidean distances are employed to evaluate the pairwise HD similarities by default, as in [1]. Other distances can be used in the HD space by specifying the dm_hds parameter. Euclidean distances are employed in the LD embedding. 
    """
    global dr_nitmax, dr_gtol, dr_ftol, dr_maxls, dr_maxcor, n_eps_np_float64, module_name
    # Checking the value of dm_hds
    dm_hds_none = dm_hds is None
    if dm_hds_none:
        dsm_hds = None
        if X_hds is None:
            raise ValueError("Error in function mssne_implem of module {module_name}: X_hds and dm_hds cannot both be None.".format(module_name=module_name))
    else:
        dsm_hds = dm_hds**2
        dsm_hds = dsm_hds.astype(np.float64)
    # Defining the random state
    rand_state = mssne_manage_seed(seed_mssne)
    # Number of data points
    if dm_hds_none:
        N = X_hds.shape[0]
    else:
        N = dsm_hds.shape[0]
    if fit_U:
        arr_ones = np.ones(shape=N, dtype=np.int64)
    # Product of N and n_components
    prod_N_nc = N*n_components
    # Maximum number of L-BFGS steps at each stage of the multi-scale optimization.
    nit_max = dr_nitmax
    # Tolerance for the norm of the gradient in the L-BFGS algorithm
    gtol = dr_gtol
    # Tolerance for the relative update of the objective function value.
    ftol = dr_ftol
    # Smallest float
    np_eps = n_eps_np_float64
    # Function to compute the gradient of the objective.
    fct_grad = mssne_grad
    # Function to compute the objective value
    fct_obj = mssne_obj_fct
    
    # Maximum number of line search steps per L-BFGS iteration.
    maxls = dr_maxls
    # Maximum number of variable metric corrections used to define the limited memory matrix in L-BFGS. 
    maxcor = dr_maxcor
    
    # Defining K_star for the multi-scale perplexities, following the notations of [2]. 
    K_star = 2
    # Computing the multi-scale perplexities, following the notations of [2]. 
    L, K_h = ms_perplexities(N=N, K_star=K_star)
    
    # Initializing the LD embedding.
    X_lds = init_lds(X_hds=X_hds, N=N, init=init, n_components=n_components, rand_state=rand_state)
    
    # Computing the multi-scale HD bandwidths if fit_U is True, and the HD similarities at the different scales otherwise. We also evaluate the LD bandwidths and precisions.
    retv_mssne_sim = mssne_sim_hds_bandwidth(X_hds=X_hds, K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U, dsm_hds=dsm_hds)
    
    if fit_U:
        dsm_hds, tau_hi, p_h, t_h = retv_mssne_sim
        dsm_hds_min_row_dsm_lds_t = mssne_eval_dsm_lds_min_row_dsm_lds_t_dsm(dsm_lds=dsm_hds)
    else:
        sim_hij_allh = retv_mssne_sim
        p_h, t_h = ms_ld_bandwidths(tau_hi=np.empty(shape=(L, N), dtype=np.float64), K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U)[2:]
    # Note that t_h >= np_eps as ensured in function ms_ld_bandwidths
    
    # Reshaping X_lds as the optimization functions work with 1-D arrays.
    X_lds = np.reshape(a=X_lds, newshape=prod_N_nc)
    
    # Matrix storing the multi-scale HD similarities sigma_{ij}. Element [i,j] contains sigma_{ij}. sigma_{ii} is set to 0. We need to recompute them progressively by adding perplexities one at the time as we perform multi-scale optimization.
    sigma_ij = np.zeros(shape=(N,N), dtype=np.float64)
    
    # Multi-scale optimization. n_perp is the number of currently considered perplexities.
    for n_perp in range(1, L+1, 1):
        # Index of the currently added perplexity.
        h = L-n_perp
        # LD precisions
        cur_p_h = p_h[h:]
        # LD bandwidths
        cur_t_h = t_h[h:]
        
        # Computing the N**2 HD similarities sigma_{hij} if fit_U is True, using the bandwidths tau_hi which were already computed. Otherwise, we just need to gather them.
        if fit_U:
            sigma_hij = mssne_lds_similarities_h(arr_den_s_i=tau_hi[h,:], np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_hds_min_row_dsm_lds_t)
        else:
            sigma_hij = sim_hij_allh[h,:,:]
        # Updating the multi-scale HD similarities
        sigma_ij = (sigma_ij*(np.float64(n_perp)-1.0) + sigma_hij)/np.float64(n_perp)
        
        # Defining the arguments of the L-BFGS algorithm
        args = (sigma_ij, N, n_components, cur_p_h, cur_t_h, n_perp)
        
        # Running L-BFGS
        res = scipy.optimize.minimize(fun=fct_obj, x0=X_lds, args=args, method='L-BFGS-B', jac=fct_grad, bounds=None, callback=None, options={'disp':False, 'maxls':maxls, 'gtol':gtol, 'maxiter':nit_max, 'maxcor':maxcor, 'maxfun':np.inf, 'ftol':ftol})
        X_lds = res.x
    
    # Reshaping the result
    X_lds = np.reshape(a=X_lds, newshape=(N, n_components))
    
    # Returning
    if ret_sim_hds:
        return X_lds, sigma_ij
    else:
        return X_lds

def mssne_sev_implem(X_hds_sev, init='random', n_components=2, fit_U=False, dm_fct=None, seed_mssne=None):
    """
    Similar to mssne_implem, but minimizes the mean cost function computed over several data sets.
    This function can be used to compute a LD embedding of an incomplete data set by minimizing the expectation of the multi-scale SNE cost function, which is estimated thanks to multiple imputations, as detailed in [1]. 
    In:
    - X_hds_sev: 3-D numpy array with shape (n_samp, N, M). For i in range(n_samp), X_hds_sev[i,:,:] contains a HD data set, with one example per row and one dimension per column, and without duplicated examples. X_hds_sev cannot be None.
    - init: determines the initialization of the LD embedding. 
    ---> If isinstance(init, str) is True:
    ------> If init is equal to 'pca', the LD embedding is initialized with the first n_components principal components of np.mean(a=X_hds_sev, axis=0).
    ------> If init is equal to 'random', the LD embedding is initialized randomly, using a uniform Gaussian distribution with small variance.
    ------> Otherwise an error is raised.
    ---> If isinstance(init, np.ndarray) is True:
    ------> init must in this case be a 2-D numpy array, with N rows and n_components columns. It stores the LD positions to use for the initialization, with one example per row and one LD dimension per column. init[i,:] contains the initial LD coordinates for the i^th HD sample in X_hds_sev, which has its coordinates for the different data sets stored in X_hds_sev[:, i, :]. If init.ndim != 2 or init.shape[0] != N or init.shape[1] != n_components, an error is raised.
    ---> Otherwise, an error is raised.
    - n_components: dimension of the LDS.
    - fit_U: boolean indicating whether to fit U in the definition of the multi-scale LD similarities. If True, the U is tuned as in [2]. Otherwise, it is forced to 1. Setting fit_U to True usually tends to slightly improve DR quality at the expense of slightly increasing computation time. 
    - dm_fct: (optional) a function taking as argument a 2-D numpy array X_hds with shape (N,M) storing a HD data set and returning a 2-D np.ndarray dm_hds with shape (N,N) containing the pairwise HD distances (NOT squared) between the data points in X_hds. In particular, dm_hds[i,j] stores the HD distance (NOT squared) between X_hds[i,:] and X_hds[j,:]. This function is used to compute the pairwise HD distances (NOT squared) between the data points in the data sets in X_hds_sev. If dm_fct is None, Euclidean distance is used. An example of a valid function for the dm_fct parameter is the eucl_dist_matr one.
    - seed_mssne: seed to use for the random state. Check mssne_manage_seed for a description. 
    Out:
    A tuple with:
    - A 2-D numpy.ndarray X_lds with shape (N, n_components), containing the LD data set, with one example per row and one dimension per column. X_lds[i,:] contains the LD coordinates related to the i^th HD sample in X_hds_sev, which has its coordinates for the different data sets stored in X_hds_sev[:, i, :].
    - sigma_ij: a 2-D numpy array with shape (N, N), containing the mean multi-scale HD similarities over the n_samp data sets in X_hds_sev.
    Remarks:
    - L-BFGS algorithm is used, as suggested in [2].
    - Multi-scale optimization is performed, as presented in [2].
    - Euclidean distances are employed to evaluate the pairwise HD similarities by default, as in [1]. Other distances can be used in the HD space by specifying the dm_fct parameter. Euclidean distances are employed in the LD embedding. 
    """
    global dr_nitmax, dr_gtol, dr_ftol, dr_maxls, dr_maxcor, n_eps_np_float64
    # Defining the random state
    rand_state = mssne_manage_seed(seed_mssne)
    # Number of data sets and number of data points per data set
    n_samp, N = X_hds_sev.shape[:2]
    # Product of N and n_components
    prod_N_nc = N*n_components
    # Maximum number of L-BFGS steps at each stage of the multi-scale optimization.
    nit_max = dr_nitmax
    # Tolerance for the norm of the gradient in the L-BFGS algorithm.
    gtol = dr_gtol
    # Tolerance for the relative update of the objective function value. 
    ftol = dr_ftol
    # Smallest float
    np_eps = n_eps_np_float64
    
    if fit_U:
        arr_ones = np.ones(shape=N, dtype=np.int64)
        # Function to compute the gradient of the objective.
        fct_grad = mssne_sev_grad
        # Function to compute the objective function
        fct_obj = mssne_sev_obj_fct
    else:
        # Function to compute the gradient of the objective.
        fct_grad = mssne_grad
        # Function to compute the objective function
        fct_obj = mssne_obj_fct
    
    # Maximum number of line search steps per L-BFGS iteration.
    maxls = dr_maxls
    # Maximum number of variable metric corrections used to define the limited memory matrix of L-BFGS.
    maxcor = dr_maxcor
    
    # Defining K_star for the multi-scale perplexities, following the notations of [2]. 
    K_star = 2
    # Computing the multi-scale perplexities, following the notations of [2]. 
    L, K_h = ms_perplexities(N=N, K_star=K_star)
    
    # Initializing the LD embedding.
    X_lds = init_lds(X_hds=np.mean(a=X_hds_sev, axis=0) if (isinstance(init, str) and (init == 'pca')) else None, N=N, init=init, n_components=n_components, rand_state=rand_state)
    
    # Computing the multi-scale HD and LD bandwidths and precisions if fit_U is True. Otherwise, the expected HD similarities at each scale over all data sets are computed, as well as the LD bandwidths and precisions.
    retv_sim_hds = mssne_sev_sim_hds_bandwidth(X_hds_sev=X_hds_sev, K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U, dm_fct=dm_fct)
    if fit_U:
        dsm_hds, tau_hi, p_h, t_h = retv_sim_hds
        dsm_hds_min_row_dsm_lds_t = mssne_sev_eval_dsm_lds_min_row_dsm_lds_t_dsm(dsm_lds=dsm_hds)
    else:
        sim_hij_allh, p_h, t_h = retv_sim_hds
    # Note that t_h >= np_eps as ensured by function ms_ld_bandwidths
    
    # Reshaping X_lds as the optimization functions work with 1-D arrays.
    X_lds = np.reshape(a=X_lds, newshape=prod_N_nc)
    
    if fit_U:
        # 3-D numpy array with shape (n_samp, N, N). For each k in range(n_samp), sigma_ij[k,:,:] is a matrix storing the multi-scale HD similarities sigma_{ij} for the data set X_hds_sev[k,:,:]. Element [k,i,j] contains sigma_{ij}. sigma_{ii} is set to 0. We need to recompute them progressively by adding perplexities one at the time as we perform multi-scale optimization, from [2]. 
        sigma_ij = np.zeros(shape=(n_samp,N,N), dtype=np.float64)
    else:
        # Matrix storing the multi-scale HD similarities sigma_{ij}. Element [i,j] contains sigma_{ij}. sigma_{ii} is set to 0. We need to recompute them progressively by adding perplexities one at the time as we perform multi-scale optimization, from [2].
        sigma_ij = np.zeros(shape=(N,N), dtype=np.float64)
    
    # Multi-scale optimization. n_perp is the number of currently considered perplexities.
    for n_perp in range(1, L+1, 1):
        # Index of the currently added perplexity.
        h = L-n_perp
        
        if fit_U:
            # LD precisions for all data sets
            cur_p_h = p_h[:,h:]
            # LD bandwidths for all data sets
            cur_t_h = t_h[:,h:]
            # Computing the N**2 HD similarities sigma_{hij} for each data set. We use the bandwidths tau_hi which were already computed.
            sigma_hij = mssne_sev_lds_similarities_h(arr_den_s_i=tau_hi[:,h,:], np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_hds_min_row_dsm_lds_t)
        else:
            # LD precisions
            cur_p_h = p_h[h:]
            # LD bandwidths
            cur_t_h = t_h[h:]
            # Gathering the expected HD similarities at scale h over the different data sets.
            sigma_hij = sim_hij_allh[h,:,:]
        
        # Updating the multi-scale HD similarities
        sigma_ij = (sigma_ij*(np.float64(n_perp)-1.0) + sigma_hij)/np.float64(n_perp)
        
        # Defining the arguments of the L-BFGS algorithm
        args = (sigma_ij, N, n_components, cur_p_h, cur_t_h, n_perp)
        # Running L-BFGS
        res = scipy.optimize.minimize(fun=fct_obj, x0=X_lds, args=args, method='L-BFGS-B', jac=fct_grad, bounds=None, callback=None, options={'disp':False, 'maxls':maxls, 'gtol':gtol, 'maxiter':nit_max, 'maxcor':maxcor, 'maxfun':np.inf, 'ftol':ftol})
        X_lds = res.x
    
    if fit_U:
        # Computing the mean of the multi-scale HD similarities over the different data sets.
        sigma_ij = sigma_ij.sum(axis=0)/np.float64(n_samp)
    
    # Returning the reshaped result as well as sigma_ij
    return np.reshape(a=X_lds, newshape=(N, n_components)), sigma_ij

##############################
############################## 
# Gaussian mixture modeling of a complete or incomplete HD data set. 
# The main functions are 'gmm_fit_K', 'gmm_sev_em_fitting' and 'gmm_sev_sampling'. 
# See their documentations for details. 
# The 'mssne_na_mmg' function presents how to use these functions. 
####################

# Boolean. If True, the tresh used in HDDC is fixed to 10**(-3) times the trace of the covariance. Otherwise, several tresholds are tested, which are returned by function tresh_values, and the one yielding the smallest BIC is selected.
only_one_tresh = False
# Tolerance for the stopping criterion of the EM algorithm.
Llog_tol = 10.0**(-5.0)
# Maximum number of iterations for the EM algorithm.
max_iter_em = 400

@numba.jit(nopython=True)
def bic(Llog, P, N):
    """
    Evaluates the BIC of a model.
    In: 
    - Llog: log-likelihood of the model.
    - P: number of parameters of the model.
    - N: number of data points.
    Out: 
    np.inf if N<P, and a float being the BIC otherwise.
    """
    if N<P:
        return np.inf
    else:
        # Casting the variables
        Llog, N, P = np.float64(Llog), np.float64(N), np.float64(P)
        # Returning
        return -2.0*Llog+P*np.log(N)

@numba.jit(nopython=True)
def sweep(G, m):
    """
    Implements the sweep operator on a symmetric matrix.
    In:
    - G: 2-D numpy array which stores a symmetric matrix.
    - m: integer between 0 and G.shape[0] which indicates the row and column over which to sweep.
    Out:
    A 2-D numpy array containing G swept on row and column m.
    """
    # Intermediate computations.
    inv_gmm = 1.0/G[m, m]
    gm = G[:, m]*inv_gmm
    # General formula
    H = G - np.outer(gm, G[m, :])
    # Modify the m-th row and column
    H[:, m], H[m, :] = gm, gm
    # Modify the (m,m) entry
    H[m, m] = -inv_gmm
    return H

@numba.jit(nopython=True)
def reverse_sweep(G,m):
    """
    Implements the reverse sweep operator on a symmetric matrix.
    In:
    - G: two-dimensional numpy array which stores a symmetric matrix.
    - m: integer between 0 and G.shape[0] which indicates the row and column over which to reverse sweep.
    Out:
    A two-dimensional numpy array containing G reversely swept on row and column m.
    """
    # Intermediate computations.
    inv_gmm = -1.0/G[m, m]
    gm = G[:, m]*inv_gmm
    # General formula
    H = G+np.outer(gm, G[m, :])
    # Modify the m-th row and column
    H[:, m], H[m, :] = gm, gm
    # Modify the (m,m) entry
    H[m, m] = inv_gmm
    return H

@numba.jit(nopython=True)
def sev_sweep(G, arr_m):
    """
    Sweeping a symmetric matrix over a set of indexes. The order does not matter as the sweep operator is commutative.
    Index:
    - G: two-dimensional numpy array which stores a symmetric matrix.
    - arr_m: one-dimensional numpy array of integers which are all between 0 and G.shape[0].
    Out:
    A two-dimensional numpy array containing G swept on the indexes in arr_m.
    """
    # Casting
    G, arr_m = G.astype(np.float64), arr_m.astype(np.int64)
    # Sweeping over each index in arr_m. The order does not matter as the sweep operator is commutative.
    for m in arr_m:
        G = sweep(G=G, m=m)
    # Returning
    return G

@numba.jit(nopython=True)
def sev_reverse_sweep(G, arr_m):
    """
    Reversely sweeping a symmetric matrix over a set of indexes. The order does not matter as the reverse sweep operator is commutative.
    Index:
    - G: two-dimensional numpy array which stores a symmetric matrix.
    - arr_m: one-dimensional numpy array of integers which are all between 0 and G.shape[0].
    Out:
    A two-dimensional numpy array containing G reversely swept on the indexes in arr_m.
    """
    # Casting
    G, arr_m = G.astype(np.float64), arr_m.astype(np.int64)
    # Reversely sweeping over each index in arr_m. The order does not matter as the reverse sweep operator is commutative.
    for m in arr_m:
        G = reverse_sweep(G=G, m=m)
    # Returning
    return G

@numba.jit(nopython=True)
def mmg_hddc_num_param(M, K, d):
    """
    Evaluates the number of parameters of a Gaussian mixture model using the HDDC model [a_ij,b_i,Q_i,d_i], introduced in [10].
    In:
    - M: integer indicating the dimension of the data points in the data set.
    - K: integer indicating the number of Gaussian mixture components.
    - d: one-dimensional numpy array of integers with size K in which the kth element contains the number of dimensions preserved for the kth Gaussian.
    Out:
    The number of parameters of the underlying Gaussian mixture model using the HDDC model [a_ij,b_i,Q_i,d_i], introduced in [10].
    """
    # Casting to float64
    M, K = np.float64(M), np.float64(K)
    d = d.astype(np.float64)
    # Defining intermediate quantities introduced in [10].
    rho = K*M+K-1
    tau = np.dot(d, M-(d+1.0)*0.5)
    D = np.sum(d)
    # Returning
    return np.int64(rho+tau+2*K+D)

@numba.jit(nopython=True)
def eval_sample_mean(X, X_mask, N):
    """
    Evaluate the sample mean of each feature in a data set with missing values.
    In:
    - X: A two-dimensional numpy array of dtype np.float64 with one example per row and one feature per column. The missing values in X are assumed to have been filled with zero's.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - N: an integer being the number of rows in X and X_mask.
    Out:
    A one-dimensional numpy array with as many elements as columns in X and X_mask, containing the sample mean of the features in X.
    """
    arr_ones = np.ones(shape=N, dtype=np.float64)
    return np.dot(arr_ones, X)/(np.float64(N)-np.dot(arr_ones, X_mask.astype(np.float64)))

@numba.jit(nopython=True)
def mmg_init_mu_cov(X, X_mask, K):
    """
    Initializing the means, covariances and mixing coefficients of a Gaussian mixture. The initialization is similar to the one in [8], which is detailed in [13], page 225.
    In:
    - X: A two-dimensional numpy array of dtype np.float64 with one example per row and one feature per column. The missing values in X are assumed to have been filled with zero's.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - K: strictly positive integer indicating the number of mixture components. An error is raised if K<=0.
    Out:
    A tuple with:
    - mu: a two-dimensional numpy array with shape (K, X.shape[1]). mu[k,:] contains the initial mean of the k^th Gaussian.
    - S: a three-dimensional numpy array with shape (K, X.shape[1], X.shape[1]). The initial covariance matrix for Gaussian k can be accessed through S[k,:,:].
    - mix_coeff: a one-dimensional numpy array with K elements. The element at index k contains the mixing coefficient associated to the kth Gaussian. These are initialized using uniform values.
    - n_ex_missing: a one-dimensional numpy array with N elements, where N is the number of rows of X. Element at index i gives the number of missing data of the ith example.
    Remarks:
    - All the Gaussian covariances are initialized to the same matrix. If there are at least M+1 complete examples, the covariance is initialized by their sample covariance (=complete-case analysis). This provides a consistent estimate of the covariance if the data are MCAR [13] (see page 225). Otherwise it is initialized as a diagonal matrix with the variance of each feature on its diagonal. Computing the covariance for each pair of feature independently (= available-case analysis) would allow to consider all the non-missing entries for each pair of features independently, and not to only rely on the examples with no missing values, but could yield an estimate which is not positive definite [13].
    - The normalization of the initial covariance matrices is (N - ddof), where N is the number of observation. ddof = 1 leads to unbiased estimates, while ddof = 0 leads to maximum likelihood estimates. As the updates of the covariances matrices in the EM algorithm are done by using ddof = 0 (maximum likelihood estimates), we initialize the covariance matrices using ddof=0, for consistency. This is indeed suggested in [8].
    - To initialize the means, examples without missing values are drawn. 
    - When there are not enough examples without missing values in order to initialize the means, the examples with the less missing values are preferred. In order to avoid always choosing the same examples, a random shuffling of the examples is performed.
    """
    if K <= 0:
        raise ValueError("Error in function mmg_init_mu_cov of module dr_missing_data.py: K must be a >0.")
    N, M = X.shape
    # Computing the number of missing values per example. We need to use dot because of numba (i.e. the line "n_ex_missing = np.sum(a=X.mask, axis=1)" does not work) and as dot preserves the data type, we cast to integers. Furthermore, dot only support float and complex arrays. This is why we first cast its arguments to float.
    n_ex_missing = (np.dot(X_mask.astype(np.float64), np.ones(shape=M, dtype=np.float64))).astype(np.int64)
    # Indexes of the examples without missing values
    idx_ex_full = np.nonzero(n_ex_missing==0)[0]
    n_ex_full = idx_ex_full.size
    # If there are enough samples without missing values, the means of the Gaussians are initialized by sampling among them. Otherwise, the samples without missing values are preferred and the remaining Gaussian means are initialized with incomplete samples where the missing values are imputed by the sample mean [8].
    if n_ex_full >= K:
        # Sampling the indexes of the examples on which the means of the Gaussians will be initialized.
        idx_mean = np.random.choice(a=idx_ex_full, size=K, replace=False)
        # Initializing the means of the K Gaussians
        mu = X[idx_mean,:]
    else:
        # Making sure that the same examples are not always chosen to initialize the means.
        ind = np.arange(N)
        np.random.shuffle(ind)
        # Indexes of the data points used to initialize the Gaussian means. These data points are the ones with the less missing values. For ties, a random sampling is performed, thanks to the above np.random.shuffle.
        ind_mu = ind[np.argsort(n_ex_missing[ind])[:K]]
        # Initializing the means and storing their mask
        mu = np.copy(X[ind_mu,:])
        mu_mask = np.copy(X_mask[ind_mu,:])
        # Indexes of the features of mu with missing values. dot only support float and complex arrays. This is why we first cast its arguments to float and then cast the result to int.
        idx_mu_feat_missing = np.nonzero((np.dot(np.ones(shape=K, dtype=np.float64), mu_mask.astype(np.float64))).astype(np.int64)>0)[0]
        # Computing the sample mean of the features indexed in idx_mu_feat_missing. 
        if idx_mu_feat_missing.size > 0:
            sample_mean = eval_sample_mean(X=X[:,idx_mu_feat_missing], X_mask=X_mask[:,idx_mu_feat_missing], N=N)
            # For each row of mu, replacing its missing values with the corresponding sample mean.
            for k in range(K):
                v_tmp = mu[k,:]
                v_tmp[np.nonzero(mu_mask[k,:])[0]] = sample_mean[np.nonzero(mu_mask[k,:][idx_mu_feat_missing])[0]]
                mu[k,:] = v_tmp
    # Initializing the covariance matrix of the Gaussians. All the Gaussian covariances are initialized to the same matrix. If there are at least M+1 complete examples, the covariance is initialized by their sample covariance. Otherwise it is initialized as a diagonal matrix with the variance of each feature on its diagonal.
    if n_ex_full >= M+1:
        X_full = X[idx_ex_full,:]
        n_ex_full_float = np.float64(n_ex_full)
        X_mu_full = X_full-np.dot(np.ones(shape=n_ex_full, dtype=np.float64), X_full)/n_ex_full_float
        S_base = np.dot(X_mu_full.T, X_mu_full)/n_ex_full_float
    else:
        # We use a loop as the following line does not work with numba: S_base = np.diag(v=X.var(axis=0, ddof=0.0), k=0)
        S_base = np.zeros(shape=(M,M), dtype=np.float64)
        for m in range(M):
            S_base[m,m] = np.var(X[np.nonzero(np.logical_not(X_mask[:,m]))[0],:][:,m])
    # Replicating the covariance matrix for each Gaussian of the mixture. The covariance matrix of the k^th Gaussian can be accessed through S[k-1,:,:]. We use a loop instead of the following line as it does not work with numba: S = np.tile(A=S_base, reps=(K,1,1)).
    S = np.empty(shape=(K,M,M), dtype=np.float64)
    for k in range(K):
        S[k,:,:] = S_base
    # Initializing the mixing coefficients of the mixture
    mix_coeff = np.ones(shape=K, dtype=np.float64)/np.float64(K)
    # Returning
    return mu, S, mix_coeff, n_ex_missing

@numba.jit(nopython=True)
def mg_cov(V, w):
    """
    Evaluates the covariance matrix of a multivariate Gaussian based on its eigenvectors and its eigenvalues, which must all be strictly positive.
    In:
    - V: two-dimensional numpy array containing the matrix containing the eigenvectors of the covariance matrix in its column. If S is the covariance matrix and D is a diagonal matrix in which the i^th diagonal element is the eigenvalue related to the eigenvector in V[:,i], than we must have S = np.dot(np.dot(V,S), V.T).
    - w: one-dimensional numpy array. Element at index i must contain the eigenvalue associated to the eigenvector V[:,i]. All the eigenvalues must be strictly positive.
    Out:
    A two-dimensional numpy array containing the covariance matrix.
    """
    return np.dot(V*w, V.T)

@numba.jit(nopython=True)
def mg_inv_cov(V, w):
    """
    Evaluates the inverse of the covariance matrix of a multivariate Gaussian based on its eigenvectors and its eigenvalues, which must all be strictly positive.
    In:
    - V: two-dimensional numpy array containing the matrix containing the eigenvectors of the covariance matrix in its column. If S is the covariance matrix and D is a diagonal matrix in which the i^th diagonal element is the eigenvalue related to the eigenvector in V[:,i], than we must have S = np.dot(np.dot(V,S), V.T).
    - w: one-dimensional numpy array. Element at index i must contain the eigenvalue associated to the eigenvector V[:,i]. All the eigenvalues must be strictly positive.
    Out:
    A two-dimensional numpy array containing the inverse of the covariance matrix.
    """
    return np.dot(V/w, V.T)

@numba.jit(nopython=True)
def mg_sweep_mat(x, mu, S):
    """
    Defining the matrix G (following the notations of [8]) on which the sweep operator will be applied with the index of the observed components in order to find the inverse of the submatrix of the covariance associated with the observed features of the current example.
    In: 
    - x: one-dimensional numpy array. M denotes x.size. The missing values in X are assumed to have been filled with zero's.
    - mu: one-dimensional numpy array of the same size as x and which contains the mean of the multivariate Gaussian.
    - S: two-dimensional numpy array with shape (mu.size, mu.size) containing the covariance matrix of the multivariate Gaussian. It must be symmetric.
    Out:
    A two-dimensional numpy array containing the G matrix, in the notations of [8].
    """
    # Substracting x from mu
    mu_x = mu-x
    # Returning
    M = x.size
    return np.vstack((np.hstack((S, mu_x.reshape((M, 1)))), np.hstack((mu_x, np.zeros(shape=1, dtype=np.float64))).reshape((1,M+1))))

@numba.jit(nopython=True)
def mg_reverse_sweep_mat(x, mu, S_inv):
    """
    Defining the corrected matrix G' (following the notations of [8], with a minus before S_inv) on which the reverse sweep operator will be applied with the index of the missing components in order to find the inverse of the submatrix of the covariance associated with the observed features of the current example.
    In: 
    - x: one-dimensional numpy array. M denotes x.size. The missing values in X are assumed to have been filled with zero's.
    - mu: one-dimensional numpy array of the same size as x and which contains the mean of the multivariate Gaussian.
    - S_inv: two-dimensional numpy array with shape (mu.size, mu.size) containing the inverse of the covariance matrix of the multivariate Gaussian. It must be symmetric.
    Out:
    A two-dimensional numpy array containing the corrected G' matrix, in the notations of [8], with a minus before S_inv.
    """
    # Substracting x from mu
    mu_x = mu-x
    # Computing the product between S_inv and mu_x
    S_inv_mu_x = np.dot(S_inv, mu_x)
    # Multiplying again by mu_x
    mu_x_S_inv_mu_x = np.empty(shape=1, dtype=np.float64)
    mu_x_S_inv_mu_x[0] = -np.dot(mu_x, S_inv_mu_x)
    # Returning
    M = x.size
    return np.vstack((np.hstack((-S_inv, S_inv_mu_x.reshape((M, 1)))), np.hstack((S_inv_mu_x, mu_x_S_inv_mu_x)).reshape((1,M+1))))

@numba.jit(nopython=True)
def mg_pdf(a, det_S_oo, n_obs):
    """
    Evaluates the marginal pdf of a multivarivate Gaussian over the observed values of some example.
    In: 
    - a: the a variable in section 2.5 of [8], equal to twice the argument of the exponential in the pdf of the marginal multivariate Gaussian over the observed values of the current example.
    - det_S_oo: Determinant of the submatrix over the observed feature of the current example of the covariance matrix of the multivariate Gaussian
    - n_obs: number of observed features of the currently considered example.
    Out:
    A float being the marginal pdf of the multivarivate Gaussian over the observed values of the current example.
    """
    # Casting
    a, det_S_oo, n_obs = np.float64(a), np.float64(det_S_oo), np.float64(n_obs)
    # Returning
    return np.exp(a*0.5)/(np.sqrt(det_S_oo)*((2.0*np.float64(np.pi))**(n_obs*0.5)))

@numba.jit(nopython=True)
def mg_pdf_complete(x, mu, S_inv, det_S):
    """
    Evaluates the pdf of a multivarivate Gaussian.
    In: 
    - x: one-dimensional numpy array containing the features of the examples. M denotes the size of x.
    - mu: one-dimensional numpy array with M elements storing the mean of the Gaussian.
    - S_inv: two-dimensional numpy array with shape (M, M) storing the inverse of the covariance matrix of the Gaussian.
    - det_S: float storing the determinant of S.
    Out:
    A float being the marginal pdf of the multivarivate Gaussian.
    """
    # Casting
    det_S = np.float64(det_S)
    S_inv, x, mu = S_inv.astype(np.float64), x.astype(np.float64), mu.astype(np.float64)
    # Returning
    xmu = x-mu
    return np.exp(-np.dot(xmu,np.dot(S_inv,xmu))*0.5)/(np.sqrt(det_S)*((2.0*np.float64(np.pi))**(np.float64(x.size)*0.5)))

@numba.jit(nopython=True)
def gmm_pdf_complete(x, mu, S_inv, det_S, mix_coeff):
    """
    Evaluates the pdfs of the Gaussian components of a mixture model.
    In:
    - x: one-dimensional numpy array containing the features of the examples. M denotes the size of x.
    - mu: a two-dimensional numpy array with shape (K, M), where K is the number of mixture components. mu[k,:] contains the mean of the k^th Gaussian.
    - S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of the covariance matrix of the kth Gaussian.
    - det_S:  one-dimensional numpy array with K elements. For each k in range(K), det_S[k] contains a float being the determinant of the covariance of the kth Gaussian.
    - mix_coeff: a one-dimensional numpy array with K elements, in which element at index k contains the mixing coefficient of the kth Gaussian. The elements in mix_coeff must be positive and their sum must be equal to 1.
    Out:
    A tuple with:
    - gmm_pdf_x: a one-dimensional numpy array with K elements, in which element at index k contains mix_coeff[k] times the pdf of the k^th Gaussian, divided by the sum over j of mix_coeff[j] times the pdf of the j^th Gaussian.
    - gmm_pdf_x_sum: a float being the sum over j of mix_coeff[j] times the pdf of the j^th Gaussian.
    """
    # Casting
    S_inv, x, mu, mix_coeff, det_S = S_inv.astype(np.float64), x.astype(np.float64), mu.astype(np.float64), mix_coeff.astype(np.float64), det_S.astype(np.float64)
    # Number of components in the mixture
    K = mix_coeff.size
    # One-dimensional numpy array with K elements, storing the pdf of the K Gaussians.
    gmm_pdf_x = np.empty(shape=K, dtype=np.float64)
    # For each mixture component
    for k in range(K):
        gmm_pdf_x[k] = mg_pdf_complete(x=x, mu=mu[k,:], S_inv=S_inv[k,:,:], det_S=det_S[k])*mix_coeff[k]
    # Summing gmm_pdf_x
    gmm_pdf_x_sum = np.sum(gmm_pdf_x)
    # Normalizing gmm_pdf_x
    gmm_pdf_x /= gmm_pdf_x_sum
    # Returning
    return gmm_pdf_x, gmm_pdf_x_sum

@numba.jit(nopython=True)
def mg_cond_mean_cov(x, x_mask, mu, S, S_inv):
    """
    Computing the conditional mean and covariance of the missing values for some example on some Gaussian component.
    In: 
    - x: one-dimensional numpy array. M denotes x.size. The missing values in x are assumed to have been filled with zero's.
    - x_mask: A one-dimensional array of boolean with the same shape as x. A True value in the mask indicates a missing data.
    - mu: one-dimensional numpy array of the same size as x and which contains the mean of the multivariate Gaussian.
    - S: two-dimensional numpy array with shape (mu.size, mu.size) containing the covariance matrix of the multivariate Gaussian. It must be symmetric.
    - S_inv: two-dimensional numpy array with shape (mu.size, mu.size) containing the inverse of the covariance matrix of the multivariate Gaussian. It must be symmetric.
    Out:
    A tuple with:
    - S_cond_mm: two-dimensional numpy array with the conditional covariance of the missing values of the current example.
    - mu_cond_m: one-dimensional numpy array with the conditional mean of the missing values of the current example.
    - marg_pdf_oo: A float being the marginal pdf of the multivarivate Gaussian over the observed values of the current example.
    - idx_feat_missing: one-dimensional numpy array of integers with the indexes of the missing features in the current example.
    Remark:
    - As the updates of the covariances matrices in the EM algorithm are done by using ddof = 0.0 (maximum likelihood estimates), we initialize the covariance matrices using ddof=0.0, for consistency. This is indeed suggested in [8].
    """
    # Casting
    S_inv, mu, S = S_inv.astype(np.float64), mu.astype(np.float64), S.astype(np.float64)
    # Dimension of x
    M = x.size
    # Indexes of the missing and observed features in x
    idx_feat_missing, idx_feat_obs = np.nonzero(x_mask)[0], np.nonzero(np.logical_not(x_mask))[0]
    # Number of missing and observed values
    n_missing, n_obs = idx_feat_missing.size, idx_feat_obs.size
    # Sweeping along the observed features if there are more missing data than observed ones, and on the missing ones otherwise.
    if n_missing > n_obs:
        H = sev_sweep(G=mg_sweep_mat(x=x, mu=mu, S=S), arr_m=idx_feat_obs)
    else:
        H = sev_reverse_sweep(G=mg_reverse_sweep_mat(x=x, mu=mu, S_inv=S_inv), arr_m=idx_feat_missing)
    # Gathering the submatrices in H which are useful
    S_cond_mm = H[idx_feat_missing,:][:,idx_feat_missing]
    mu_cond_m = H[idx_feat_missing,M]
    # Evaluating the marginal pdf of the multivarivate Gaussian over the observed values of the current example.
    marg_pdf_oo = mg_pdf(a=H[M, M], det_S_oo=np.linalg.det(S[idx_feat_obs,:][:,idx_feat_obs]), n_obs=n_obs)
    # Returning
    return S_cond_mm, mu_cond_m, marg_pdf_oo, idx_feat_missing

@numba.jit(nopython=True)
def mmg_cond_mean_cov(x, x_mask, n_mis, mu, S, S_inv, S_cond, mix_coeff):
    """
    Computing the conditional mean and covariance of the missing values for some example on all the Gaussian components of the mixture.
    In:
    - x: one-dimensional numpy array. M denotes x.size. The missing values in x are assumed to have been filled with zero's.
    - x_mask: A one-dimensional array of boolean with the same shape as x. A True value in the mask indicates a missing data.
    - n_mis: integer equal to the number of missing values in x.
    - mu: a two-dimensional numpy array with shape (K, M), where K is the number of mixture components. mu[k,:] contains the mean of the k^th Gaussian.
    - S: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S[k,:,:] contains a two-dimensional numpy array storing the covariance matrix of the kth Gaussian.
    - S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of the covariance matrix of the kth Gaussian.
    - S_cond: a three-dimensional numpy array with shape (K, M, M). S_cond[k,:,:] will be updated, for each k in range(K), as S_cond[k,idx_feat_missing,:][:,:,idx_feat_missing] += S_cond_mm[k,:,:]*marg_pdf_oo[k], where idx_feat_missing is a one-dimensional numpy array of integers with the indexes of the missing features in x, S_cond_mm[k,:,:] is a two-dimensional numpy array with shape (n_mis, n_mis) containing the conditional covariance of the missing values of x with respect to the kth Gaussian, and marg_pdf_oo[k] is equal to the product of mix_coeff[k] and the marginal pdf of the kth Gaussian over the observed values of x, divided by the sum over j in range(K) of the product of mix_coeff[j] and the marginal pdf of the jth Gaussian over the observed values of x.
    - mix_coeff: a one-dimensional numpy array with K elements, in which element at index k contains the mixing coefficient of the kth Gaussian. The elements in mix_coeff must be positive and their sum must be equal to 1.
    Out:
    A tuple with:
    - mu_cond_m: two-dimensional numpy array with shape (K,n_mis). The k^th row contains the conditional mean of the missing values of x using the k^th Gaussian.
    - marg_pdf_oo: one-dimensional numpy array with K elements. marg_pdf_oo[k] is equal to the product of mix_coeff[k] and the marginal pdf of the kth Gaussian over the observed values of x, divided by the sum over j in range(K) of the product of mix_coeff[j] and the marginal pdf of the jth Gaussian over the observed values of x.
    - marg_pdf_oo_sum: a float being equal to the sum over j in range(K) of the product of mix_coeff[j] and the marginal pdf of the jth Gaussian over the observed values of x.
    - idx_feat_missing: a one-dimensional numpy array with n_mis integers with the indexes of the missing features in x
    Moreover, S_cond is updated as S_cond[k,idx_feat_missing,:][:,:,idx_feat_missing] += S_cond_mm[k,:,:]*marg_pdf_oo[k] for each k in range(K)
    """
    # Casting
    S_inv, mu, S, S_cond, mix_coeff = S_inv.astype(np.float64), mu.astype(np.float64), S.astype(np.float64), S_cond.astype(np.float64), mix_coeff.astype(np.float64)
    n_mis = np.int64(n_mis)
    # Number of Gaussian components
    K = np.int64(mu.shape[0])
    # One-dimensional numpy array with K elements, which will store the marginal pdf of the multivarivate Gaussians over the observed values of the current example.
    marg_pdf_oo = np.empty(shape=K, dtype=np.float64)
    # Two-dimensional numpy array with shape (K,n_mis). The k^th row will contain the conditional mean of the missing values in x using the k^th Gaussian.
    mu_cond_m = np.empty(shape=(K,n_mis), dtype=np.float64)
    # Three-dimensional numpy array with shape (K,n_mis,n_mis). S_cond_mm[k,:,:] will contain the conditional covariances of the missing values of the current example on the kth Gaussian.
    S_cond_mm = np.empty(shape=(K,n_mis,n_mis), dtype=np.float64)
    # For each Gaussian component
    for k in range(K):
        # Evaluating the conditional mean and covariance of missing values of the current example on the current Gaussian
        S_cond_mm[k,:,:], mu_cond_m[k,:], marg_pdf_oo[k], idx_feat_missing = mg_cond_mean_cov(x=x, x_mask=x_mask, mu=mu[k,:], S=S[k,:,:], S_inv=S_inv[k,:,:])
    # Multiplying the marginal pdf of each Gaussian of the observed values of the current example by the mixing coefficients of the Gaussian.
    marg_pdf_oo *= mix_coeff
    # Summing marg_pdf_oo
    marg_pdf_oo_sum = np.sum(marg_pdf_oo)
    # Normalizing marg_pdf_oo
    marg_pdf_oo /= marg_pdf_oo_sum
    # Updating S_cond. We use a loop as the following line does not work with numba: S_cond[:,idx_feat_missing,:][:,:,idx_feat_missing] += S_cond_mm*marg_pdf_oo[:,np.newaxis,np.newaxis]
    for k in range(K):
        # We use this trick with v_tmp and w_tmp to update S_cond as allocating to S_cond[k,:,:][idx_feat_missing,:][:,idx_feat_missing] does not modify S_cond.
        v_tmp = S_cond[k,:,:]
        w_tmp = v_tmp[idx_feat_missing,:]
        w_tmp[:,idx_feat_missing] += S_cond_mm[k,:,:]*marg_pdf_oo[k]
        v_tmp[idx_feat_missing,:] = w_tmp
        S_cond[k,:,:] = v_tmp
    # Returning
    return mu_cond_m, marg_pdf_oo, marg_pdf_oo_sum, idx_feat_missing

@numba.jit(nopython=True)
def mmg_cond_mean_cov_sample_mm(x, x_mask, n_mis, mu, S, S_inv, mix_coeff):
    """
    Computing the conditional mean and covariance of the missing values of some example over all the Gaussian components of the mixture.
    In:
    - x: one-dimensional numpy array. M denotes x.size. The missing values in x are assumed to have been filled with zero's.
    - x_mask: A one-dimensional array of boolean with the same shape as x. A True value in the mask indicates a missing data.
    - n_mis: integer equal to the number of missing values in x.
    - mu: a two-dimensional numpy array with shape (K, M), where K is the number of mixture components. mu[k,:] contains the mean of the k^th Gaussian.
    - S: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S[k,:,:] contains a two-dimensional numpy array storing the covariance matrix of the kth Gaussian.
    - S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of the covariance matrix of the kth Gaussian.
    - mix_coeff: a one-dimensional numpy array with K elements, in which element at index k contains the mixing coefficient of the kth Gaussian. The elements in mix_coeff must be positive and their sum must be equal to 1.
    Out:
    A tuple with:
    - mu_cond_m: two-dimensional numpy array with shape (K,n_mis). The k^th row contains the conditional mean of the missing values of x using the k^th Gaussian.
    - S_cond_mm: three-dimensional numpy array with shape (K,n_mis,n_mis). S_cond_mm[k,:,:] contains the conditional covariances of the missing values of the current example on the kth Gaussian.
    - marg_pdf_oo: one-dimensional numpy array with K elements. marg_pdf_oo[k] is equal to the product of mix_coeff[k] and the marginal pdf of the kth Gaussian over the observed values of x, divided by the sum over j in range(K) of the product of mix_coeff[j] and the marginal pdf of the jth Gaussian over the observed values of x.
    - idx_feat_missing: a one-dimensional numpy array with n_mis integers with the indexes of the missing features in x
    """
    # Casting
    S_inv, mu, S, mix_coeff = S_inv.astype(np.float64), mu.astype(np.float64), S.astype(np.float64), mix_coeff.astype(np.float64)
    n_mis = np.int64(n_mis)
    # Number of Gaussian components
    K = np.int64(mu.shape[0])
    # One-dimensional numpy array with K elements, which will store the marginal pdf of the multivariate Gaussians over the observed values of the current example.
    marg_pdf_oo = np.empty(shape=K, dtype=np.float64)
    # Two-dimensional numpy array with shape (K,n_mis). The k^th row will contain the conditional mean of the missing values in x using the k^th Gaussian.
    mu_cond_m = np.empty(shape=(K,n_mis), dtype=np.float64)
    # Three-dimensional numpy array with shape (K,n_mis,n_mis). S_cond_mm[k,:,:] will contain the conditional covariances of the missing values of the current example on the kth Gaussian.
    S_cond_mm = np.empty(shape=(K,n_mis,n_mis), dtype=np.float64)
    # For each Gaussian component
    for k in range(K):
        # Evaluating the conditional mean and covariance of the missing values of the current example on the current Gaussian
        S_cond_mm[k,:,:], mu_cond_m[k,:], marg_pdf_oo[k], idx_feat_missing = mg_cond_mean_cov(x=x, x_mask=x_mask, mu=mu[k,:], S=S[k,:,:], S_inv=S_inv[k,:,:])
    # Multiplying the marginal pdf of each Gaussian of the observed values of the current example by the mixing coefficients of the Gaussians.
    marg_pdf_oo *= mix_coeff
    # Normalizing marg_pdf_oo
    marg_pdf_oo /= np.sum(marg_pdf_oo)
    # Returning
    return mu_cond_m, S_cond_mm, marg_pdf_oo, idx_feat_missing

@numba.jit(nopython=True)
def mg_hddc(S, tresh=-1):
    """
    Apply the HDDC [a_ij, b_i, Q_i, d_i] model from [10] to a Gaussian component from a mixture.
    A boolean is returned indicating whether an error appeared or not. If yes, the computation must be aborted. For example, some eigenvalues may be negative, or they can all be close to zero.
    In:
    - S: two-dimensional numpy array storing the covariance matrix of the current Gaussian component. It must be symmetric, its eigenvalues must all be >= 0 and at least one must be >0. If one of these last two conditions is not satisfied, an error is raised.
    - tresh: a floating treshold to select the dimension d according to the scree test [14]. The eigenvalues which are exactly preserved are the d first ones. The dimension d is the smallest positive integer such that the difference between the (d+1)^th and (d+2)^th eigenvalues is smaller than tresh times the trace of the covariance matrix, when the eigenvalues are sorted in descending order. By convention, the (M+1)th and (M+2)^th "eigenvalues" of S are equal to the Mth one, where M=S.shape[0]. If tresh is <0, it is set to 0.001. tresh should not be too close to 1 to avoid ill-conditioned covariance matrices, in which case the boolean value returned by this function will be set to False.
    Out:
    A tuple with:
    - A tuple with:
    ---> S: a two-dimensional numpy array containing the update of S corresponding to the [a_ij, b_i, Q_i, d_i] model from [10]. S is ensured to be symmetric positive definite. 
    ---> S_inv: a two-dimensional numpy array containing the inverse of S.
    ---> d: an integer indicating the selected dimension. d is between 0 and M-1, where M=S.shape[0].
    ---> det_S: a float being the determinant of S.
    - all_ok: a boolean. If True, the computation can go on. If False, an error occured and the computation must abort. For example, some eigenvalues may be negative, or they can all be close to zero, or there are inf values or nan in S.
    """
    # Casting
    S = S.astype(np.float64)
    # Checking that the elements of S are finite
    if not np.all(np.isfinite(S)):
        # There are some nan or inf values. The computation must be aborted.
        return (S, S, 0, 0.0), False
    # Computing the EVD of S. w is a one-dimensional numpy array with its eigenvalues sorted in ascending order. V has the corresponding eigenvectors in its columns. We have that np.all(np.isclose(S, np.dot(V*w, V.T))) is True.
    w, V = np.linalg.eigh(S)
    w, V = w.astype(np.float64), V.astype(np.float64)
    # Checking that there are no negative eigenvalue.
    if (w[0]<0) and (not close_to_zero(w[0])):
        # There are some negative eigenvalues. The computation must be aborted.
        return (S, S, 0, 0.0), False
    # Checking that all the eigenvalues are not close to zero
    if close_to_zero(w[-1]):
        # The biggest eigenvalue of S is close to 0. The computation must be aborted
        return (S, S, 0, 0.0), False
    # If the tresh on the difference between two subsequent eigenvalues is < 0, fix tresh_v to 0.001 times the trace of the covariance.
    tr_cov = np.sum(w)
    if tresh < 0:
        tresh_v = 0.001*tr_cov
    else:
        tresh_v = tresh*tr_cov
    # Computing the differences between the successive eigenvalues.
    w_diff = np.diff(w, 1)
    # Indexes of the elements of w_diff which are smaller than tresh_v
    idx_wd_tresh = np.nonzero(w_diff<tresh_v)[0]
    # Finding the index idx_chg such that the eigenvalues in w[:idx_chg] must be replaced with their mean.
    if idx_wd_tresh.size == 0:
        idx_chg = 1
    else:
        idx_chg = idx_wd_tresh[-1]+2
    # Ensuring that the eigenvalues in w[:idx_chg] are not all equal to 0.
    while close_to_zero(w[idx_chg-1]):
        idx_chg += 1
    # Changing eigenvalues in w[:idx_chg] to their mean
    w[:idx_chg] = np.mean(w[:idx_chg])
    # Recording the number of eigenvalues which remain unchanged
    d = w.size - idx_chg
    # Returning S, its inverse, the number of eigenvalues which are exactly preserved and the determinant of S
    return (mg_cov(V=V, w=w), mg_inv_cov(V=V, w=w), d, np.prod(w)), True

@numba.jit(nopython=True)
def mmg_hddc(S, tresh=-1):
    """
    Apply the HDDC [a_ij, b_i, Q_i, d_i] model from [10] to each Gaussian component of a mixture.
    A boolean is returned, indicating whether an error was produced in mg_hddc and whether the computation must be aborted.
    In:
    - S: a three-dimensional numpy array with shape (K, M, M), where K is the number of mixture components and M the dimension. For each k in range(K), S[k,:,:] must contain a two-dimensional numpy array storing the covariance matrix of the kth Gaussian component. It must be symmetric, its eigenvalues must all be >= 0 and there must be at least one which is >0.
    - tresh: see tresh parameter of function mg_hddc.
    Out:
    A tuple with:
    - A tuple with:
    ---> S: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S[k,:,:] contains a two-dimensional numpy array containing the update of covariance of the kth Gaussian component corresponding to the [a_ij, b_i, Q_i, d_i] model from [10]. S[k,:,:] is ensured to be symmetric positive definite. 
    ---> S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of S[k,:,:].
    ---> d: a one-dimensional numpy array with K elements. For each k in range(K), d[k] contains an integer indicating the selected dimension of the kth Gaussian. d[k] is between 0 and M-1.
    ---> det_S:  one-dimensional numpy array with K elements. For each k in range(K), det_S[k] contains a float being the determinant of S[k,:,:].
    - all_ok: a boolean. If True, the algorithm can go on. Otherwise, an error was produced in mg_hddc and the computation must be aborted.
    """
    # Casting
    S = S.astype(np.float64)
    # Number of components in the mixture and dimensions
    K, M = S.shape[:2]
    # Allocating space to save the inverses of the covariance of each Gaussian.
    S_inv = np.empty(shape=(K,M,M), dtype=np.float64)
    # Allocating space to save the numbers of eigenvalues kept unchanged in HDDC for each Gaussian.
    d = np.empty(shape=K, dtype=np.int64)
    # Allocating space to save the determinant of the covariance for each Gaussian.
    det_S = np.empty(shape=K, dtype=np.float64)
    # For each Gaussian in the mixture
    for k in range(K):
        ret_v, all_ok = mg_hddc(S=S[k,:,:], tresh=tresh)
        if all_ok:
            S[k,:,:], S_inv[k,:,:], d[k], det_S[k] = ret_v
        else:
            return (S, S_inv, d, det_S), False
    # Returning
    return (S, S_inv, d, det_S), True

@numba.jit(nopython=True)
def gmm_loglikelihood(marg_pdf_oo_mix):
    """
    Evaluate the marginal log-likelihood of the Gaussian mixture model given the observed features in the data set.
    N denotes the number of examples in the data set.
    In: 
    - marg_pdf_oo_mix: one-dimensional numpy array with N elements, in which element i contains the sum of the marginal pdf of the observed features of the ith example for each Gaussian in the mixture, weighted by the mixing coefficients.
    Out:
    A float being equal to np.log(marg_pdf_oo_mix).sum().
    """
    # Casting
    marg_pdf_oo_mix = marg_pdf_oo_mix.astype(np.float64)
    # Returning
    return np.sum(np.log(marg_pdf_oo_mix))

@numba.jit(nopython=True)
def gmm_e_step(X, X_mask, mu, S, S_inv, det_S, mix_coeff, ex_mis, n_ex_mis):
    """
    Computes the E-step of the EM algorithm for fitting a Gaussian mixture model on a data set with missing data. N denotes the number of data points.
    In:
    - X: A two-dimensional numpy array with one example per row and one feature per column. The missing values in X are assumed to have been filled with zero's.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - mu: a two-dimensional numpy array with shape (K, M), where K is the number of mixture components. mu[k,:] contains the mean of the k^th Gaussian.
    - S: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S[k,:,:] contains a two-dimensional numpy array storing the covariance matrix of the kth Gaussian.
    - S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of the covariance matrix of the kth Gaussian.
    - det_S: one-dimensional numpy array with K elements. For each k in range(K), det_S[k] contains a float being the determinant of the covariance of the kth Gaussian.
    - mix_coeff: a one-dimensional numpy array with K elements, in which element at index k contains the mixing coefficient of the kth Gaussian. The elements in mix_coeff must be positive and their sum must be equal to 1.
    - ex_mis: one-dimensional numpy array with X.shape[0] elements in which element at index i is True if X[i,:] has missing values, False otherwise.
    - n_ex_mis: one-dimensional numpy array with X.shape[0] elements in which element at index i is indicates the number of missing values in X[i,:].
    Out:
    A tuple with:
    - t_ki: two-dimensional numpy array with shape (K,N). Element at index [k,i] contains t_ik, following the notations of [8], section 2.2.
    - X_filled: Three-dimensional numpy array with shape (K,N,M). X_filled[k,i,:] contains X[i,:], but in which the missing values are replaced by their conditional mean under the kth Gaussian.
    - S_cond: Three-dimensional numpy array with shape (K,M,M). S_cond[k,:,:] contains the sum over i in range(N) of t_ik*\tilde{S}_ik, following the notations of [8], section 2.2.
    - Llog: float indicating the log-likelyhood of the current mixture model
    """
    # Casting
    S_inv, mu, S, mix_coeff, det_S = S_inv.astype(np.float64), mu.astype(np.float64), S.astype(np.float64), mix_coeff.astype(np.float64), det_S.astype(np.float64)
    # Number of examples and dimension
    N, M = X.shape
    # Number of Gaussian components
    K = mu.shape[0]
    # Two-dimensional numpy array with shape (K,N). Element at index [k,i] will contain t_ik, following the notations of [8], section 2.2.
    t_ki = np.empty(shape=(K,N), dtype=np.float64)
    # Three-dimensional numpy array with shape (K,N,M). X_filled[k,i,:] will contain X[i,:], but in which the missing values are replaced with their conditional mean under the kth Gaussian. 
    X_filled = np.empty(shape=(K,N,M), dtype=np.float64)
    for k in range(K):
        X_filled[k,:,:] = X
    # Three-dimensional numpy array with shape (K,M,M). S_cond[k,:,:] will contain the sum over i in range(N) of t_ik*\tilde{S}_ik, following the notations of [8], section 2.2. 
    S_cond = np.zeros(shape=(K,M,M), dtype=np.float64)
    # One-dimensional numpy array with N elements, in which element i contains the sum of the marginal pdf of the observed features of the ith example for each Gaussian in the mixture, weighted by the mixing coefficients.
    marg_pdf_oo_mix = np.empty(shape=N, dtype=np.float64)
    # For each example
    for i in range(N):
        # Checking whehter the ith example has missing values or not.
        if ex_mis[i]:
            # Updating X_filled, t_ki, marg_pdf_oo_mix and S_cond by managing X[i,:], which has missing values.
            mu_cond_m, t_ki[:,i], marg_pdf_oo_mix[i], idx_feat_missing = mmg_cond_mean_cov(x=X[i,:], x_mask=X_mask[i,:], n_mis=n_ex_mis[i], mu=mu, S=S, S_inv=S_inv, S_cond=S_cond, mix_coeff=mix_coeff)
            # We loop as X_filled[:,i,idx_feat_missing] = mu_cond_m is not allowed in numba.
            for k in range(K):
                # We use this trick with v_tmp as directly allocating X_filled[k,i,:][idx_feat_missing] does not modify X_filled
                v_tmp = X_filled[k,i,:]
                v_tmp[idx_feat_missing] = mu_cond_m[k,:]
                X_filled[k,i,:] = v_tmp
        else:
            # Updating t_ki and marg_pdf_oo_mix by managing X[i,:], which has no missing values.
            t_ki[:,i], marg_pdf_oo_mix[i] = gmm_pdf_complete(x=X[i,:], mu=mu, S_inv=S_inv, det_S=det_S, mix_coeff=mix_coeff)
    # Evaluating the log-likelyhood of the current mixture model
    Llog = gmm_loglikelihood(marg_pdf_oo_mix=marg_pdf_oo_mix)
    # Returning
    return t_ki, X_filled, S_cond, Llog

@numba.jit(nopython=True)
def gmm_m_step(t_ki, X_filled, S_cond, tresh):
    """
    Computes the M-step of the EM algorithm for fitting a Gaussian mixture model on a data set with missing data. N denotes the number of data points, K the number of Gaussian components and M the dimension.
    A boolean is returned, indicating whether an error appeared in mg_hddc and whether the computation must be aborted.
    In: 
    - t_ki: two-dimensional numpy array with shape (K,N). Element at index [k,i] contains t_ik, following the notations of [8], section 2.2.
    - X_filled: Three-dimensional numpy array with shape (K,N,M). X_filled[k,i,:] contains X[i,:], but in which the missing values are replaced with their conditional mean under the kth Gaussian.
    - S_cond: Three-dimensional numpy array with shape (K,M,M). S_cond[k,:,:] contains the sum over i in range(N) of t_ik*\tilde{S}_ik, following the notations of [8], section 2.2.
    - tresh: see tresh parameter of function mg_hddc.
    Out: 
    A tuple with:
    - A tuple with:
    ---> mu: a two-dimensional numpy array with shape (K, M), where K is the number of mixture components. mu[k,:] contains the mean of the k^th Gaussian.
    ---> S: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S[k,:,:] contains a two-dimensional numpy array storing the covariance matrix of the kth Gaussian.
    ---> S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of the covariance matrix of the kth Gaussian.
    ---> d: a one-dimensional numpy array with K elements. For each k in range(K), d[k] contains an integer d_k indicating the HDDC selected dimension of the kth Gaussian, with d_k as defined in [10]. d[k] is between 0 and M-1.
    ---> det_S: one-dimensional numpy array with K elements. For each k in range(K), det_S[k] contains a float being the determinant of the covariance of the kth Gaussian.
    ---> mix_coeff: a one-dimensional numpy array with K elements, in which element at index k contains the mixing coefficient of the kth Gaussian. The elements in mix_coeff must be positive and their sum must be equal to 1.
    - all_ok: a boolean. If True, the algorithm can go on. Otherwise, an error was produced in mg_hddc and the computation must be aborted.
    """
    # Casting
    t_ki, X_filled, S_cond = t_ki.astype(np.float64), X_filled.astype(np.float64), S_cond.astype(np.float64)
    # Number of mixture components, of examples and of dimensions.
    K, N, M = X_filled.shape
    # One-dimensional numpy array with K elements, where K is the number of mixture components. The element at index k stores N_k, following the notations of [8].
    N_k = np.dot(t_ki, np.ones(N, np.float64))
    # Evaluating the mean of the Gaussian components. We use a loop as the following way to compute the mean do not work with numba: mu = (np.sum(a=t_ki[:,:,np.newaxis]*X_filled, axis=1, dtype=np.float64).T/N_k).T
    mu = np.zeros((M, K), np.float64)
    for k in range(K):
        mu[:,k] = np.dot(X_filled[k,:,:].T, t_ki[k,:])
    mu = (mu/N_k).T
    # Computing the covariance matrices of the K Gaussians
    S = np.empty(shape=(K,M,M), dtype=np.float64)
    for k in range(K):
        X_mu_k = X_filled[k,:,:] - mu[k,:]
        S[k,:,:] = (np.dot(X_mu_k.T*t_ki[k,:], X_mu_k) + S_cond[k,:,:])/N_k[k]
    # Evaluating the mixing coefficients
    mix_coeff = N_k/float(N)
    # Applying HDDC to the covariances of the Gaussians
    ret_v, all_ok = mmg_hddc(S=S, tresh=tresh)
    if all_ok:
        S, S_inv, d, det_S = ret_v
    else:
        return (mu, S, S, mix_coeff.astype(np.int64), mix_coeff, mix_coeff), False
    # Returning
    return (mu, S, S_inv, d, det_S, mix_coeff), True

@numba.jit(nopython=True)
def gmm_em_fitting(X, X_mask, K, tresh=-1):
    """
    Fit a Gaussian mixture model with K components on a data set with missing data by using the EM algorithm.
    In: 
    - X: A two-dimensional numpy array with one example per row and one feature per column. The missing values in X are assumed to have been filled with zero's.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - K: strictly positive integer indicating the number of mixture components.
    - tresh: see description of tresh parameter of mg_hddc.
    Out:
    A tuple with:
    - model_bic: BIC of the model, or np.inf if its number of parameters is > than X.shape[0].
    - n_param: integer indicating the number of parameters of the model.
    - mu: a two-dimensional numpy array with shape (K, M), where K is the number of mixture components. mu[k,:] contains the mean of the k^th Gaussian.
    - S: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S[k,:,:] contains a two-dimensional numpy array storing the covariance matrix of the kth Gaussian.
    - mix_coeff: a one-dimensional numpy array with K elements, in which element at index k contains the mixing coefficient of the kth Gaussian. The elements in mix_coeff must be positive and their sum must be equal to 1.
    - S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of the covariance matrix of the kth Gaussian.
    - d: a one-dimensional numpy array with K elements. For each k in range(K), d[k] contains an integer d_k indicating the HDDC selected dimension of the kth Gaussian, with d_k as defined in [10]. d[k] is between 0 and M-1.
    - det_S: one-dimensional numpy array with K elements. For each k in range(K), det_S[k] contains a float being the determinant of the covariance of the kth Gaussian.
    - X_cond_mean_imp: X but in which the missing values have been imputed by their conditional mean.
    - nit: an integer indicating the number of EM iterations which have been performed.
    - max_it_done: a boolean indicating whether or not the maximum number of iteration has been reached.
    - L_converg: a boolean indicating whether the log-likelihood of the model has converged or not.
    - ex_mis: one-dimensional numpy array with X.shape[0] elements in which element at index i is True if X[i,:] has missing values, False otherwise.
    - n_ex_mis: one-dimensional numpy array with X.shape[0] elements in which element at index i is indicates the number of missing values in X[i,:].
    Remarks:
    - The EM iterations are stopped as soon as the log-likelihood does not evolve anymore or as the maximum number of iterations is reached. The set of parameters leading to the highest found value of the log-likelihood is returned.
    - The normalization of the initial covariance matrices is by (N - ddof), where N is the number of observation. ddof = 1.0 leads to unbiased estimates, while ddof = 0.0 leads to maximum likelihood estimates. As the updates of the covariances matrices in the EM algorithm are done using ddof = 0.0 (maximum likelihood estimates), we initialize the covariance matrices using ddof=0.0, for consistency. This is indeed suggested in [8].
    """
    global Llog_tol, max_iter_em
    # Maximum number of iterations of the EM algorithm.
    max_iter = max_iter_em
    # Initializing the means, covariances and mixing coefficients of the components of the mixture. n_ex_mis is a one-dimensional numpy array with X.shape[0] elements in which element at index i indicates the number of missing values in X[i,:]. The initialization is repeated until mmg_hddc does not produce errors.
    all_ok = False
    while not all_ok:
        mu, S, mix_coeff, n_ex_mis = mmg_init_mu_cov(X=X, X_mask=X_mask, K=K)
        # One-dimensional numpy array with X.shape[0] elements in which element at index i is True if X[i,:] has missing values, False otherwise.
        ex_mis = (n_ex_mis>0)
        # Applying HDDC to the initial covariances
        ret_v, all_ok = mmg_hddc(S=S, tresh=tresh)
        if all_ok:
            S, S_inv, d, det_S = ret_v
    # Iteration counter
    nit = 0
    # Log-Likelihood of the mixture model at the previous EM iteration
    Llog_prev = -np.inf
    # Highest log-likelihood currently found
    Llog_best = -np.inf
    # Alternating the E- and M-steps of the EM algorithm.
    while True: 
        # Computing the E-step
        t_ki, X_filled, S_cond, Llog = gmm_e_step(X=X, X_mask=X_mask, mu=mu, S=S, S_inv=S_inv, det_S=det_S, mix_coeff=mix_coeff, ex_mis=ex_mis, n_ex_mis=n_ex_mis)
        # Updating the parameters yielding the highest log-likelihood currently found
        if (Llog > Llog_best) or (nit == 0):
            t_ki_best, X_filled_best, Llog_best, mu_best, S_best, S_inv_best, d_best, det_S_best, mix_coeff_best = np.copy(a=t_ki), np.copy(a=X_filled), Llog, np.copy(a=mu), np.copy(a=S), np.copy(a=S_inv), np.copy(a=d), np.copy(a=det_S), np.copy(a=mix_coeff)
        # If the Log-likelihood does not increase anymore or the maximum number of iteration is reached, stop. Otherwise, perform the M-step.
        if (np.absolute(Llog-Llog_prev)<=Llog_tol) or (nit >= max_iter):
            break
        else:
            # Computing the M-step
            ret_v_m_step, all_ok = gmm_m_step(t_ki=t_ki, X_filled=X_filled, S_cond=S_cond, tresh=tresh)
            if all_ok:
                mu, S, S_inv, d, det_S, mix_coeff = ret_v_m_step
            else:
                # An error was produced during the m-step. Stopping the iterations.
                break
        # Incrementing the iteration counter
        nit += 1
        # Updating Llog_prev
        Llog_prev = Llog
    # Log-likelihood at the last iteration
    Llog_last = Llog
    # Gathering the parameters yielding the highest log-likelihood found
    t_ki, X_filled, Llog, mu, S, S_inv, d, det_S, mix_coeff = t_ki_best, X_filled_best, Llog_best, mu_best, S_best, S_inv_best, d_best, det_S_best, mix_coeff_best
    # Computing the number of parameters of the model
    N, M = X.shape
    n_param = mmg_hddc_num_param(M=M, K=K, d=d)
    # Evaluating the BIC of the model
    model_bic = bic(Llog=Llog, P=n_param, N=N)
    # Computing X but where the missing values are replaced with their conditional mean. We use a loop instead of the following line as numba does not accept it: X_cond_mean_imp = np.sum(a=(t_ki[:,:,np.newaxis]*X_filled), axis=0)
    X_cond_mean_imp = np.zeros(shape=(M,N), dtype=np.float64)
    for k in range(K):
        X_cond_mean_imp += X_filled[k,:,:].T*t_ki[k,:]
    X_cond_mean_imp = X_cond_mean_imp.T
    # Whether or not the maximum number of iterations was performed
    max_it_done = (nit >= max_iter)
    # Whether or not the log-likelihood stopped evolving
    L_converg = (np.absolute(Llog_last-Llog_prev)<=Llog_tol)
    # Returning
    return model_bic, n_param, mu, S, mix_coeff, S_inv, d, det_S, X_cond_mean_imp, nit, max_it_done, L_converg, ex_mis, n_ex_mis

@numba.jit(nopython=True)
def tresh_values():
    """
    The different tresholds to consider when using the HDDC algorithm. The treshold yielding the model with the smallest BIC will be selected.
    Out:
    A one-dimensional numpy array, sorted in decreasing order, in which each element is a valid tresh argument for function gmm_em_fitting.
    Remarks: 
    - The tresholds should not be too close to 1 to avoid ill-conditioned covariance matrices, in which case mg_hddc will indicate an error during its execution.
    - The returned array must be sorted in decreasing order.
    """
    global only_one_tresh
    if only_one_tresh:
        return -np.ones(shape=1, dtype=np.float64)
    else:
        # Returning
        return 2.0**(-np.arange(15).astype(np.float64))

@numba.jit(nopython=True)
def gmm_sev_em_fitting(X, X_mask, K, n_fit=10, seed=-1):
    """
    Similar to gmm_em_fitting but compute it n_fit times and returns the model with the smallest BIC, using the tresh parameter leading to the smallest BIC, according to the following heuristic: the function tresh_values() returns a one-dimensional numpy array of tresholds, sorted in decreasing order. Imagine that we have considered all tresholds before index t of the array, and that the best model we discovered has a BIC equal to best_bic. Then n_fit models are created using the threshold at index t+1 and best_bic is updated if one of these has a smaller BIC than best_bic. If best_bic has decreased, then the treshold at index t+2 will be considered. Otherwise the search is stopped and the subsequent thresholds will not be considered, as they are smaller and thus yield more complex models, with more parameters.
    In:
    - X, X_mask, K: see gmm_em_fitting
    - n_fit: strictly positive integer indicating the number of times the EM algorithm is run. The best model over the n_fit ones is the one having the smallest BIC.
    - seed: integer to set the random seed. If >=0, fix the np.random seed. Otherwise, it does nothing.
    Out:
    A tuple with:
    - The tuple returned by function gmm_em_fitting describing the best of n_fit models, using the best tresh value considered in function tresh_values, leading to the smallest BIC according to the above heuristic.
    - The best treshold found according to the BIC and the above heuristic, among the values returned by function tresh_values.
    """
    # Using double precision floats.
    X = X.astype(np.float64)
    # Fixing the np.random seed if the seed parameter is >= 0.
    if seed >= 0:
        np.random.seed(seed)
    # Checking n_fit
    if n_fit <= 0:
        raise ValueError("Error in function gmm_sev_em_fitting of module dr_missing_data.py: n_fit must be a strictly positive integer.")
    # Gathering the different treshold values
    all_tresh = tresh_values()
    n_tresh = all_tresh.size
    # Float storing the BIC of the best model currently found
    best_bic = np.inf
    # Float storing the BIC of the best model currently found using the previously explored treshold.
    best_bic_prev_tresh = np.inf
    # For each tresh
    for id_tresh in range(n_tresh):
        tresh = all_tresh[id_tresh]
        # Fitting n_fit models
        for i in range(n_fit):
            res_gmm = gmm_em_fitting(X=X, X_mask=X_mask, K=K, tresh=tresh)
            model_bic = res_gmm[0]
            if (model_bic < best_bic) or ((id_tresh == 0) and (i == 0)):
                best_res, best_bic, best_tresh = res_gmm, model_bic, tresh
        # If the BIC did not decreased, do not create the model for the subsequent tresholds.
        if best_bic < best_bic_prev_tresh:
            best_bic_prev_tresh = best_bic
        else:
            break
    # Returning
    return best_res, best_tresh

@numba.jit(nopython=True)
def gmm_fit_K(X, X_mask, seed=-1, n_fit=10):
    """
    Fit a Gaussian mixture model on a data set with missing data using the EM algorithm. The number of components in the mixture is derived heuristically as proposed in [9]: first, n_fit mixture models with 1 component are created and the one with the smallest BIC is retained. If it has more parameters than samples, a message is displayed and the model is returned. Otherwise, n_fit models with two components are created and the one with the smallest BIC is retained. If it has more parameters than samples or if its BIC is larger than the one of the retained model with 1 component, the best of n_fit models with one component is returned. Otherwise, the best of n_fit models with 2 components replaces the best of n_fit ones with 1 component and n_fit models with 3 components are created, the one with the smallest BIC is retained and compared to the best of n_fit models with 2 components. This goes on until the BIC of the best of n_fit models increases or when it has more parameters than samples.
    The tresh of the HDDC algorithm is determined as detailed in function gmm_sev_em_fitting.
    In: 
    - X: A two-dimensional numpy array with one example per row and one feature per column. The missing values in X are assumed to have been filled with zeros.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - seed: an integer to set the seed of np.random. If seed > 0, the seed of np.random is modified. If seed <= 0, the seed of np.random is not modified. 
    - n_fit: strictly positive integer indicating the number of times the EM algorithm is run for each number of mixture components. For each number K of mixture components, the best model over the n_fit ones is the one having the smallest BIC.
    Out: 
    The tuple returned by function gmm_sev_em_fitting(X, X_mask, K, n_fit), with the best K found by the above heuristic aiming to minimize the BIC and the best BIC of the n_fit models created with this K.
    """
    # Fixing the numpy RandomState
    if seed > 0:
        np.random.seed(seed)
    # Fitting n_fit mixtures of Gaussians with only one component and retaining the best one in terms of BIC
    K = 1
    res_gmm, best_tresh = gmm_sev_em_fitting(X=X, X_mask=X_mask, K=K, n_fit=n_fit)
    model_bic = res_gmm[0]
    # Checking whether or not the BIC of the best of n_fit models is inf. If yes, it means that the model has more parameters than X.shape[0]. It is then useless to fit a model with a larger K.
    if np.isinf(model_bic):
        print("Warning in function gmm_fit_K of module dr_missing_data.py: the mixture model with one component already has more parameters than samples. N=",X.shape[0],", n_param=",res_gmm[1],".")
        return res_gmm, best_tresh
    else:
        # Fitting mixture models with more components until the BIC increases or there are more parameters than samples
        while True:
            # Fitting n_fit mixtures of Gaussians and retaining the best one in terms of BIC
            K += 1
            res_gmm_new, best_tresh_new = gmm_sev_em_fitting(X=X, X_mask=X_mask, K=K, n_fit=n_fit)
            new_model_bic = res_gmm_new[0]
            # If new_model_bic is > than model_bic or if it is inf, in which case the best of n_fit models has more parameters than samples, we terminate. Otherwise, we continue with a larger K.
            if (new_model_bic >= model_bic) or np.isinf(new_model_bic):
                return res_gmm, best_tresh
            else:
                # Updating res_gmm and model_bic
                res_gmm, model_bic, best_tresh = res_gmm_new, new_model_bic, best_tresh_new

def gmm_sampling(X, X_mask, mu, S, S_inv, mix_coeff, ex_mis, n_ex_mis, rand_state=None):
    """
    Sample from a conditional mixture model to impute the missing data in a data set. The sampling from a conditional mixture model is described in section 3.2 of [9].
    In: 
    - X: A two-dimensional numpy array with one example per row and one feature per column. The missing values in X are assumed to have been filled with zeros.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - mu: a two-dimensional numpy array with shape (K, M), where K is the number of mixture components. mu[k,:] contains the mean of the k^th Gaussian.
    - S: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S[k,:,:] contains a two-dimensional numpy array storing the covariance matrix of the kth Gaussian.
    - S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of the covariance matrix of the kth Gaussian.
    - mix_coeff: a one-dimensional numpy array with K elements, in which element at index k contains the mixing coefficient of the kth Gaussian. The elements in mix_coeff must be positive and their sum must be equal to 1.
    - ex_mis: one-dimensional numpy array with X.shape[0] elements in which element at index i is True if X[i,:] has missing values, False otherwise.
    - n_ex_mis: one-dimensional numpy array with X.shape[0] elements in which element at index i indicates the number of missing values in X[i,:].
    - rand_state: numpy random state. If it is None, it is set to np.random.
    Out:
    A two-dimensional numpy array containing X but in which the missing values have been replaced with samples from the conditional Gaussian mixture model.
    """
    if rand_state is None:
        rand_state = np.random
    # Casting
    S_inv, mu, S, mix_coeff = S_inv.astype(np.float64), mu.astype(np.float64), S.astype(np.float64), mix_coeff.astype(np.float64)
    # Number of samples
    N = X.shape[0]
    # Number of mixture components
    K = mu.shape[0]
    # Indexes of the mixture components
    idx_mix_comp = np.arange(start=0, step=1, stop=K, dtype=np.int64)
    # Filling the data set
    go = True
    while go:
        # Data set to return
        X_filled = X.copy()
        # For each example
        for i in range(N):
            # If the ith example has missing values
            if ex_mis[i]:
                # Computing the conditional mean and covariance of the missing values of some example over all the Gaussian components of the mixture, as well as their conditional mixing coefficients.
                mu_cond_m, S_cond_mm, t_ik, idx_feat_missing = mmg_cond_mean_cov_sample_mm(x=X[i,:], x_mask=X_mask[i,:], n_mis=n_ex_mis[i], mu=mu, S=S, S_inv=S_inv, mix_coeff=mix_coeff)
                # Sampling the Gaussian component that will be used to fill the missing data of the ith example 
                k = rand_state.choice(a=idx_mix_comp, size=1, replace=False, p=t_ik)[0]
                # Filling the missing values in X[i,:] by drawing samples from a multivariate Gaussian with mean mu_cond_m[k,:] and covariance S_cond_mm[k,:,:].
                X_filled[i,idx_feat_missing] = rand_state.multivariate_normal(mean=mu_cond_m[k,:], cov=S_cond_mm[k,:,:])
        # Checking whether there are two identical samples in X_filled.
        go = contains_ident_ex(X=X_filled)
    # Returning
    return X_filled

def gmm_sev_sampling(X, X_mask, n_samp, mu, S, S_inv, mix_coeff, ex_mis, n_ex_mis, rand_state=None):
    """
    Create n_samp complete data sets by sampling n_samp times from a conditional mixture model to impute the missing values in X. The imputations are different from one sampling to the other.
    In: 
    - X: A two-dimensional numpy array with one example per row and one feature per column. The missing values in X are assumed to have been filled with zeros.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - n_samp: a strictly positive integer indicating the number of random samplings to perform. An error is raised if n_samp <= 0.
    - mu: a two-dimensional numpy array with shape (K, M), where K is the number of mixture components. mu[k,:] contains the mean of the k^th Gaussian.
    - S: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S[k,:,:] contains a two-dimensional numpy array storing the covariance matrix of the kth Gaussian.
    - S_inv: a three-dimensional numpy array with shape (K, M, M). For each k in range(K), S_inv[k,:,:] contains a two-dimensional numpy array storing the inverse of the covariance matrix of the kth Gaussian.
    - mix_coeff: a one-dimensional numpy array with K elements, in which element at index k contains the mixing coefficient of the kth Gaussian. The elements in mix_coeff must be positive and their sum must be equal to 1.
    - ex_mis: one-dimensional numpy array with X.shape[0] elements in which element at index i is True if X[i,:] has missing values, False otherwise.
    - n_ex_mis: one-dimensional numpy array with X.shape[0] elements in which element at index i indicates the number of missing values in X[i,:].
    - rand_state: numpy random state. If it is None, it is set to np.random.
    Out:
    A three-dimensional numpy array X_imp with shape (n_samp, X.shape[0], X.shape[1]). For each i in range(n_samp), X_imp[i,:,:] contains X but in which the missing values have been replaced with samples from the conditional Gaussian mixture model.
    """
    global module_name
    if rand_state is None:
        rand_state = np.random
    # Casting
    S_inv, mu, S, mix_coeff = S_inv.astype(np.float64), mu.astype(np.float64), S.astype(np.float64), mix_coeff.astype(np.float64)
    # Checking n_samp value
    if n_samp <= 0:
        raise ValueError("Error in function gmm_sev_sampling of module {module_name}: n_samp={n_samp} while it should be >0.".format(module_name=module_name, n_samp=n_samp))
    # Number of samples and dimensions
    N, M = X.shape
    # Data sets to return. For each i in range(n_samp), X_imp[i,:,:] will contain X but with its missing values replaced with random samples from the conditional Gaussian mixture model.
    X_imp = np.empty(shape=(n_samp, N, M), dtype=np.float64)
    # For each random sampling
    for i in range(n_samp):
        X_imp[i,:,:] = gmm_sampling(X=X, X_mask=X_mask, rand_state=rand_state, mu=mu, S=S, S_inv=S_inv, mix_coeff=mix_coeff, ex_mis=ex_mis, n_ex_mis=n_ex_mis)
    # Returning
    return X_imp

##############################
############################## 
# Methods to perform single imputation of the missing entries in a data set. 
# These methods are the ones employed in the experiments of [1] to compare their performances with respect to the results of the approach presented in [1]. 
# The main functions are 'mu_si_implem' and 'icknni_implem'. 
# See their documentations for details.
# The 'mssne_na_mu_si' and 'mssne_na_icknni' functions present how to use 'mu_si_implem' and 'icknni_implem'. 
####################

@numba.jit(nopython=True)
def mu_si_implem(X, X_mask):
    """
    Replaces the missing values in a data set with the mean of their features.
    In:    
    - X: A two-dimensional numpy array with shape (N, M), with one example per row and one feature per column. The missing values in X are assumed to have been filled with zero's. There should not be features with only missing values.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    Out:
    A two-dimensional numpy array with shape (N, M) in which the missing values of X have been replaced with the mean of their features.
    """
    # Number of samples
    N = X.shape[0]
    # Cast
    X = X.astype(np.float64)
    # Mean of the features
    mean_feat = eval_sample_mean(X=X, X_mask=X_mask, N=N)
    # For each example
    for i in range(N):
        # Indexes of the missing values of the ith example
        idx_na = np.nonzero(X_mask[i,:])[0]
        # Replacing the missing values with the mean of their features.
        for j in idx_na:
            X[i,j] = mean_feat[j]
    # Returning
    return X

@numba.jit(nopython=True)
def sqeucl_dist(X, x):
    """
    Compute the Euclidean distances between a vector and a bunch of others. 
    In:
    - X: a 2-D numpy.ndarray with one example per row and one feature per column. 
    - x: a 1-D numpy.ndarray such that x.size = X.shape[1].
    Out:
    A 1-D numpy.ndarray with X.shape[0] elements and in which element i is the squared Euclidean distance between x and X[i,:].
    """
    M = X-x
    return (M*M).sum(axis=1)

@numba.jit(nopython=True)
def icknni_implem(X, X_mask, k_nn=5, dnn_fct=sqeucl_dist):
    """
    Fill the missing values in an incomplete data set using ICkNNI [15].
    In:
    - X: A two-dimensional numpy array with shape (N, M), with one example per row and one feature per column. The missing values in X are assumed to have been filled with zeros. There should not be features with only missing values.
    - X_mask: A two-dimensional array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - k_nn: maximum number of nearest neighbors to use for the imputation. The eligible nearest neighbors are the ones with the same observed features as the currently processed example x_i, in addition to the feature j of x_i that we currently want to impute. If there are between 1 and k_nn eligible nearest neighbors, the mean of their feature j is used to impute the feature j of x_i. If there are more than k_nn eligible nearest neighbors, the mean of the feature j of the k_nn nearest neighbors is used to impute the feature j of x_i. If there are no eligible nearest neighbor, then the sample mean of the j^th feature is used to impute the j^th feature of x_i.
    - dnn_fct: distance function to use between the samples. It must be a numba jitted function with nopython = True. An example of a valid function is the sqeucl_dist one. The function must have 2 parameters, X and x. X must be a 2-D numpy.ndarray with one example per row and one feature per column, while x must be a 1-D numpy.ndarray such that x.size = X.shape[1]. The function must return a 1-D numpy.ndarray with X.shape[0] elements and in which element i is the distance between x and X[i,:]. By default, the sqeucl_dist function is employed, which is equivalent to employing a Euclidean distance. 
    Out:
    A two-dimensional numpy array with shape (N, M) in which the missing values of X have been replaced according to ICkNNI [15].
    """
    # Number of samples and dimension of X
    N, M = X.shape
    # Cast
    X = X.astype(np.float64)
    # Mean of the features
    mean_feat = eval_sample_mean(X=X, X_mask=X_mask, N=N)
    # Imputed data set to return
    X_imp = X.copy()
    # Opposite of X_mask
    X_obs = np.logical_not(X_mask)
    # For each example
    for i in range(N):
        # Indexes of the missing values of the ith example
        idx_na = np.nonzero(X_mask[i,:])[0]
        # If the current example has missing values
        if idx_na.size > 0:
            # Removing the i^th example from X and X_obs
            i_1 = i+1
            X_no_i = np.vstack((X[:i,:], X[i_1:,:]))
            X_obs_no_i = np.vstack((X_obs[:i,:], X_obs[i_1:,:]))
            # Indexes of the observed features of the ith example
            idx_obs = np.nonzero(X_obs[i,:])[0]
            n_obs = idx_obs.size
            onesf_n_obs = np.ones(shape=n_obs, dtype=np.float64)
            # Indexes in X_obs_no_i of the examples with observed features containing the ones observed in the ith example
            elig_nn = np.nonzero((np.dot(X_obs_no_i[:,idx_obs].astype(np.float64), onesf_n_obs)).astype(np.int64) == n_obs)[0]
            # For each missing feature of the ith example
            for j in idx_na:
                # Defining the eligible nearest neighbors based on the currently processed missing feature of the i^th example
                elig_nn_j = elig_nn[np.nonzero(X_obs_no_i[elig_nn,j])[0]]
                n_elig_nn_j = elig_nn_j.size
                # If there are no eligible nearest neighbor, then imputing with the sample mean. Otherwise, searching the k_nn nearest neighbors among the currently eligible ones.
                if n_elig_nn_j == 0:
                    X_imp[i,j] = mean_feat[j]
                else:
                    # Determining the k_nn nearest neighbors of the ith example among the eligible ones
                    if n_elig_nn_j > k_nn:
                        # Computing the distances to the eligible nearest neighbors
                        d_se_elig_nn_j = dnn_fct(X_no_i[elig_nn_j,:][:,idx_obs], X[i,:][idx_obs])
                        # Determining the k_nn nearest neighbors according to their distances to the i^th example.
                        nn_j = elig_nn_j[np.argsort(d_se_elig_nn_j)[:k_nn]]
                    else:
                        # All the eligible nearest neighbors are considered
                        nn_j = elig_nn_j
                    # Imputing
                    X_imp[i,j] = np.mean(X_no_i[:,j][nn_j])
    # Returning
    return X_imp

##############################
############################## 
# Methods to perform nonlinear dimensionality reduction of an incomplete data set through multi-scale SNE using single imputation of the missing data. 
# These methods are the ones employed in the experiments of [1] to compare their performances with respect to the results of the approach presented in [1]. 
# See the documentations of the 'mssne_na_mu_si' and 'mssne_na_icknni' functions for details.
# The demo at the end of this file presents how to use these functions. 
####################

def mssne_na_mu_si(X, X_mask, dim_LDS=2, init_mssne='random', dm_fct_mssne=None, fitU_mssne=False, seed_mssne=None):
    """
    Reduce the dimension of an incomplete HD data set using multi-scale SNE by managing the missing data using the Mean Imputation method, as reviewed in [1]. 
    The missing entries are hence first imputed with the mean of their features, using the mu_si_implem function. 
    Multi-scale SNE is then applied on the resulting data set. 
    In this documentation, N denotes the number of examples in the HD data set and M its dimension.
    In:
    - X: a 2-D numpy array with shape (N, M), with one example per row and one feature per column. The missing values in X are assumed to be filled with zeros. There should be no example nor feature containing only missing values. There should not be duplicated examples. 
    - X_mask: A 2-D numpy array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - dim_LDS: targeted dimension of the LDS. 
    - init_mssne: init parameter in mssne_implem. See mssne_implem for a description. In particular, init_mssne can be equal to 'pca', 'random' or a 2-D numpy array containing the initial LD coordinates of the data points. 
    - dm_fct_mssne: (optional) a function taking as argument a 2-D numpy array X_hds with shape (N,M) storing a complete HD data set and returning a 2-D np.ndarray dm_hds with shape (N,N) containing the pairwise HD distances (NOT squared) between the data points in X_hds. In particular, dm_hds[i,j] stores the HD distance (NOT squared) between X_hds[i,:] and X_hds[j,:]. This function is used to compute the pairwise HD distances (NOT squared) between the data points in the imputed data set provided to multi-scale SNE. If dm_fct_mssne is None, Euclidean distance is used in the HD space in multi-scale SNE. An example of a valid function for the dm_fct_mssne parameter is the eucl_dist_matr one.
    - fitU_mssne: fit_U parameter of the mssne_implem function. Setting it to False typically tends to decrease computation time, while setting it to True usually leads to slightly improved DR quality. 
    - seed_mssne: same as in mssne_implem. It is an integer enabling to set the random seed for multi-scale SNE. If it is >0, it sets the random seed. If it is <=0, it is ignored and the random seed is not modified. If it is None, it is treated as equal to seed_MsSNE_def. If it is not an integer, an error is raised.
    Out:
    A tuple with: 
    - X_lds: a 2-D numpy array with shape (N, dim_LDS) storing the obtained LD representation. It contains one example per row and one feature per column. Example in row i corresponds to the example in row i of X.
    - sim_hd_na: the second element of the tuple returned by mssne_implem when ret_sim_hds is set to True. 
    - t_na: the computation time to deal with the missing values, in seconds.
    - t_dr: the computation time to create the LDS, in seconds.
    """
    # Replacing the missing values with the mean of their features
    t0 = time.time()
    X_imp = mu_si_implem(X=X, X_mask=X_mask)
    t_na = time.time() - t0
    
    # Applying multi-scale SNE
    t0 = time.time()
    X_lds, sim_hd_na = mssne_implem(X_hds=X_imp, init=init_mssne, n_components=dim_LDS, ret_sim_hds=True, fit_U=fitU_mssne, seed_mssne=seed_mssne, dm_hds=None if (dm_fct_mssne is None) else dm_fct_mssne(X_imp))
    t_dr = time.time() - t0
    
    # Returning
    return X_lds, sim_hd_na, t_na, t_dr

def mssne_na_icknni(X, X_mask, k_nn=5, dim_LDS=2, init_mssne='random', dnn_fct=sqeucl_dist, dm_fct_mssne=None, fitU_mssne=False, seed_mssne=None):
    """
    Reduce the dimension of an incomplete HD data set using multi-scale SNE by managing the missing data using the ICkNNI method [15], as reported in the experiments of [1]. 
    The missing entries are hence first imputed using the icknni_implem function. 
    Multi-scale SNE is then applied on the resulting data set. 
    In this documentation, N denotes the number of examples in the HD data set and M its dimension.
    In:
    - X: a 2-D numpy array with shape (N, M), with one example per row and one feature per column. The missing values in X are assumed to be filled with zeros. There should be no example nor feature containing only missing values. There should not be duplicated examples. 
    - X_mask: A 2-D numpy array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - k_nn: k_nn parameter of the icknni_implem function. 
    - dim_LDS: targeted dimension of the LD embedding. 
    - init_mssne: init parameter in mssne_implem. See mssne_implem for a description. In particular, init_mssne can be equal to 'pca', 'random' or a 2-D numpy array containing the initial LD coordinates of the data points. 
    - dnn_fct: distance function to use in ICkNNI to find the nearest neighbors. It must be a numba jitted function with nopython = True. An example of a valid function is the sqeucl_dist one. The function must have 2 parameters, X and x. X must be a 2-D numpy.ndarray with one example per row and one feature per column, while x must be a 1-D numpy.ndarray such that x.size = X.shape[1]. The function must return a 1-D numpy.ndarray with X.shape[0] elements and in which element i is the distance between x and X[i,:]. By default, the sqeucl_dist function is employed, which is equivalent to employing a Euclidean distance. 
    - dm_fct_mssne: a function taking as argument a 2-D numpy array X_hds with shape (N,M) storing a complete HD data set and returning a 2-D np.ndarray dm_hds with shape (N,N) containing the pairwise HD distances (NOT squared) between the data points in X_hds. In particular, dm_hds[i,j] stores the HD distance (NOT squared) between X_hds[i,:] and X_hds[j,:]. This function is used to compute the pairwise HD distances (NOT squared) between the data points in the imputed data set provided to multi-scale SNE. If dm_fct_mssne is None, Euclidean distance is used in the HD space in multi-scale SNE. An example of a valid function for the dm_fct_mssne parameter is the eucl_dist_matr one.
    - fitU_mssne: fit_U parameter of the mssne_implem function. Setting it to False typically tends to decrease computation time, while setting it to True usually leads to slightly improved DR quality. 
    - seed_mssne: same as in mssne_implem. It is an integer enabling to set the random seed for multi-scale SNE. If it is >0, it sets the random seed. If it is <=0, it is ignored and the random seed is not modified. If it is None, it is treated as equal to seed_MsSNE_def. If it is not an integer, an error is raised.
    Out:
    A tuple with: 
    - X_lds: a 2-D numpy array with shape (N, dim_LDS) storing the obtained LD representation. It contains one example per row and one feature per column. Example in row i corresponds to the example in row i of X.
    - sim_hd_na: the second element of the tuple returned by mssne_implem when ret_sim_hds is set to True. 
    - t_na: the computation time to deal with the missing values, in seconds.
    - t_dr: the computation time to create the LDS, in seconds.
    """
    # Apply ICkNNI
    t0 = time.time()
    X_imp = icknni_implem(X=X, X_mask=X_mask, k_nn=k_nn, dnn_fct=dnn_fct)
    t_na = time.time() - t0
    
    # Applying multi-scale SNE
    t0 = time.time()
    X_lds, sim_hd_na = mssne_implem(X_hds=X_imp, init=init_mssne, n_components=dim_LDS, ret_sim_hds=True, fit_U=fitU_mssne, seed_mssne=seed_mssne, dm_hds=None if (dm_fct_mssne is None) else dm_fct_mssne(X_imp))
    t_dr = time.time() - t0
    
    # Returning    
    return X_lds, sim_hd_na, t_na, t_dr

##############################
############################## 
# Function to perform nonlinear dimensionality reduction of an incomplete data set using multi-scale SNE through multiple imputations or conditional mean imputation of the missing data, as presented in [1]. 
# See the documentation of the 'mssne_na_mmg' function for details.
# The demo at the end of this file presents how to use it. 
####################

def mssne_na_mmg(X, X_mask, dr_mi=True, n_mi=100, dr_si=False, fit_K=True, dim_LDS=2, init_mssne='random', dm_fct_mssne=None, n_em_fit=10, fitU_mssne=False, seed_mmg=None, seed_mssne=None):
    """
    Reduce the dimension of an incomplete HD data set using multi-scale SNE by managing the missing data using the methodology presented in [1]. 
    A Gaussian mixture is first learned on the incomplete database. Its number of components can be restricted to 1 by setting fit_K to False, in which case a single multivariate Gaussian is fitted. Otherwise, if fit_K is True, a Gaussian mixture is considered and its number of components is tuned as described in [1]. 
    Then, the dimension of the database is reduced using multi-scale SNE thanks to multiple imputations (dr_mi = True) of the missing data using the data distribution model, and/or a conditional mean imputation (dr_si = True) of the missing entries based on the data distribution model. 
    In this documentation, N denotes the number of examples (including the incomplete ones) in the HD data set and M its dimension.
    In:
    - X: a 2-D numpy array with shape (N, M), with one example per row and one feature per column. The missing values in X are assumed to be filled with zeros. There should be no example nor feature containing only missing values. There should not be duplicated examples. 
    - X_mask: A 2-D numpy array of boolean with the same shape as X. A True value in the mask indicates a missing data.
    - dr_mi: boolean. If True, multiple imputations of the missing entries in the incomplete HD data set are performed, and the expected cost function of multi-scale SNE is minimized to determine the LD embedding. 
    - n_mi: number of imputations to perform if dr_mi is True, in order to estimate the expectation of the cost function of multi-scale SNE through multiple imputations. 
    - dr_si: boolean. If True, conditional mean imputation of the missing entries in the incomplete HD data set is performed, and the dimension of the resulting data set is reduced using multi-scale SNE. 
    - fit_K: boolean. If True, a Gaussian mixture model of the incomplete data set is learned and its number of components is tuned as described in [1]. If False, a single multivariate Gaussian is tuned. Setting fit_K to True typically significantly improves the quality of the results, according to [1]. 
    - dim_LDS: targeted dimension of the LDS. 
    - init_mssne: init parameter in mssne_implem. See mssne_implem for a description. In particular, init_mssne can be equal to 'pca', 'random' or a 2-D numpy array containing the initial LD coordinates of the data points. 
    - dm_fct_mssne: (optional) a function taking as argument a 2-D numpy array X_hds with shape (N,M) storing a complete HD data set and returning a 2-D np.ndarray dm_hds with shape (N,N) containing the pairwise HD distances (NOT squared) between the data points in X_hds. In particular, dm_hds[i,j] stores the HD distance (NOT squared) between X_hds[i,:] and X_hds[j,:]. This function is used to compute the pairwise HD distances (NOT squared) between the data points in the imputed data sets provided to multi-scale SNE. If dm_fct_mssne is None, Euclidean distance is used in the HD space in multi-scale SNE. An example of a valid function for the dm_fct_mssne parameter is the eucl_dist_matr one.
    - n_em_fit: n_fit parameter of the gmm_fit_K and gmm_sev_em_fitting functions. It specifies the number of times the EM is run with different initializations for a fixed number of Gaussian components, in order to find the best possible local maximum. 
    - fitU_mssne: fit_U parameter of the mssne_implem function. Setting it to False typically tends to decrease computation time, while setting it to True usually leads to slightly improved DR quality. 
    - seed_mmg: an integer enabling to set the random seed for the distribution modeling of the incomplete data set. If it is >0, it sets the random seed. If it is <=0, it is ignored and the random seed is not modified. If it is None, it is treated as equal to 3. If it is not an integer, an error is raised.
    - seed_mssne: same as in mssne_implem function. It is an integer enabling to set the random seed for multi-scale SNE. If it is >0, it sets the random seed. If it is <=0, it is ignored and the random seed is not modified. If it is None, it is treated as equal to seed_MsSNE_def. If it is not an integer, an error is raised.
    Out:
    A tuple with: 
    - None if dr_mi is False. Otherwise, if dr_mi is True, a tuple with the results associated to the multiple imputations. The elements of the tuple are structured as follows: 
    ---> X_lds_mi: a 2-D numpy array with shape (N, dim_LDS) storing the LD embedding obtained by minimizing the expected cost function of multi-scale SNE, estimated through multiple imputations. It contains one example per row and one feature per column. Example in row i corresponds to the example in row i of X.
    ---> sim_hd_na_mi: the second element of the tuple returned by mssne_sev_implem. 
    ---> t_na_mi: the computation time to deal with the missing values, in seconds. This includes the fitting of the parametric data distribution model and the multiple imputations. 
    ---> t_dr_mi: the computation time to create the LD embedding, in seconds. This consists in minimizing the expected cost function of multi-scale SNE, estimated through multiple imputations. 
    ---> X_mi: the value returned by the gmm_sev_sampling function. It is a 3-D numpy array storing the multiple imputations of the incomplete data set X through the parametric data distribution model. 
    - None if dr_si is False. Otherwise, if dr_si is True, a tuple with the results associated to the conditional mean imputation. The elements of the tuple are structured as follows: 
    ---> X_lds_si: a 2-D numpy array with shape (N, dim_LDS) storing the LD embedding obtained by applying multi-scale SNE on the HD data set in which the missing entries have been filled through conditional mean imputation. It contains one example per row and one feature per column. Example in row i corresponds to the example in row i of X.
    ---> sim_hd_na_si: the second element of the tuple returned by mssne_implem when ret_sim_hds is set to True. 
    ---> t_na: the computation time to deal with the missing values, in seconds. This consists in the fitting of the parametric data distribution model. 
    ---> t_dr_si: the computation time to create the LD embedding, in seconds. This consists in applying multi-scale SNE on the HD data set in which the missing entries have been filled through conditional mean imputation. 
    - the tuple returned by gmm_fit_K(X=X, X_mask=X_mask, seed=seed_mmg, n_fit=n_em_fit) if fit_K is True, and the tuple returned by gmm_sev_em_fitting(X=X, X_mask=X_mask, K=1, n_fit=n_em_fit, seed=seed_mmg) if fit_K is False. 
    Remarks:
    - This function can be employed with both dr_mi and dr_si set to True. This produces 2 LD versions of the incomplete HD database, respectively resulting from the multiple imputations and the conditional mean imputation. Setting both dr_mi and dr_si to True at once is faster than running this function twice, respectively with (dr_mi, dr_si) = (True, False) and with (dr_mi, dr_si) = (False, True). Indeed, as the same Gaussian mixture model (if fit_K is True) or multivariate Gaussian model (if fit_K is False) is used for both the multiple imputations and the conditional mean imputation, the model only needs to be learned once when this function is used with dr_mi=True and dr_si=True, instead of twice if this function is called 2 times. 
    - If X has several hundreds features or more, it may be difficult to tune the data distribution model using Gaussian mixtures, as they may require to fit too many model parameters. One way to reduce the number of parameters to tune is to force fit_K to False, to restrict the parametric data distribution model to a single multivariate Gaussian. If this is not sufficient, a less general HDDC model than the one employed in [1] should be used, according to the developments presented in [10]. In this case, the 'mg_hddc' and 'mmg_hddc' functions should be modified accordingly, to implement the chosen HDDC model. 
    """
    global module_name
    
    # Checking the seed_mmg parameter
    if seed_mmg is None:
        seed_mmg = 3
    
    if seed_mmg != int(round(seed_mmg)):
        raise ValueError("Error in function mssne_na_mmg of module {module_name}: seed_mmg must be an integer.".format(module_name=module_name))
    
    if seed_mmg > 0:
        rand_state = np.random.RandomState(seed_mmg+1)
    else:
        rand_state = np.random
    
    # If fit_K is True, a Gaussian mixture model is learned on the incomplete data set, and its number of components is tuned. If fit_K is False, a single multivariate Gaussian is fitted on the incomplete data set, hence restricting the number of components of the Gaussian mixture to 1. 
    t0 = time.time()
    vg = gmm_fit_K(X=X, X_mask=X_mask, seed=seed_mmg, n_fit=n_em_fit) if fit_K else gmm_sev_em_fitting(X=X, X_mask=X_mask, K=1, n_fit=n_em_fit, seed=seed_mmg)
    t_na = time.time() - t0
    model_bic, n_param, mu, S, mix_coeff, S_inv, d, det_S, X_cond_mean_imp, nit, max_it_done, L_converg, ex_mis, n_ex_mis = vg[0]
    
    # Applying multi-scale SNE on the data set where the missing values have been replaced with their conditional means.
    if dr_si:
        t0 = time.time()
        X_lds_si, sim_hd_na_si = mssne_implem(X_hds=X_cond_mean_imp, init=init_mssne, n_components=dim_LDS, ret_sim_hds=True, fit_U=fitU_mssne, seed_mssne=seed_mssne, dm_hds=None if (dm_fct_mssne is None) else dm_fct_mssne(X_cond_mean_imp))
        t_dr_si = time.time() - t0
        
        ret_si = X_lds_si, sim_hd_na_si, t_na, t_dr_si
    else:
        ret_si = None
    
    # Creating n_mi data sets through multiple imputations and minimizing the expected cost function of multi-scale SNE using them to compute the LD embedding
    if dr_mi:
        t0 = time.time()
        X_mi = gmm_sev_sampling(X=X, X_mask=X_mask, n_samp=n_mi, rand_state=rand_state, mu=mu, S=S, S_inv=S_inv, mix_coeff=mix_coeff, ex_mis=ex_mis, n_ex_mis=n_ex_mis)
        t_na_mi = time.time() - t0 + t_na
        
        # Applying multi-scale SNE on X_mi
        t0 = time.time()
        X_lds_mi, sim_hd_na_mi = mssne_sev_implem(X_hds_sev=X_mi, init=init_mssne, n_components=dim_LDS, fit_U=fitU_mssne, seed_mssne=seed_mssne, dm_fct=dm_fct_mssne)
        t_dr_mi = time.time() - t0
        
        ret_mi = X_lds_mi, sim_hd_na_mi, t_na_mi, t_dr_mi, X_mi
    else:
        ret_mi = None
    
    # Returning
    return ret_mi, ret_si, vg

##############################
############################## 
# Function to simulate missing values in a data set. 
# See the documentation of the 'add_missing' function for details. 
# The demo at the end of this file presents how this function can be used. 
####################

def add_missing(X, p, rand_state=np.random):
    """
    Add missing values to a data set.
    In: 
    - X: a two-dimensional numpy array with one example per row and one feature per column.
    - p: a float strictly between 0 and 1 indicating the proportion of missing values in the data set.
    - rand_state: numpy random state.
    Out:
    A numpy.ma.MaskedArray with the same shape as X and in which the proportion of missing values (i.e. mask entries equal to True) is equal to p. The function avoids inducing examples or features with only missing values. An error is raised when this is not possible.
    A True value in the mask indicates a misssing entry.
    Remark:
    - An error is raised if p is such that there must be at least one feature or example with only missing values.
    """
    global module_name
    # Casting
    X, p = X.astype(dtype=np.float64, copy=False), np.float64(p)
    if (p<=0) or (p>=1):
        raise ValueError("Error in function add_missing of module {module_name}: p={p} while it should be strictly between 0 and 1.".format(module_name=module_name, p=p))
    # Shape of the data set.
    N, M = X.shape
    # Number of missing and observed values.
    N_M = N*M
    n_missing = int(round(N_M*p))
    n_obs = N_M - n_missing
    # Checking whether we can avoid that an example or a feature only has missing values.
    if (n_missing>(N*(M-1))) or (n_missing>((N-1)*M)):
        raise ValueError("Error in function add_missing of module {module_name}: p={p} induces {n_missing} missing values. As there are N={N} examples and M={M} features, there cannot be more than {max_ex} (resp. {max_feat}) missing values to avoid that an example (resp. a feature) only has missing values.".format(module_name=module_name, N=N, M=M, p=p, n_missing=n_missing, max_ex=N*(M-1), max_feat=(N-1)*M))
    # Randomly defining the elements which will be missing.
    mask_entries = np.hstack((np.zeros(shape=n_obs, dtype=np.bool), np.ones(shape=n_missing, dtype=np.bool)))
    rand_state.shuffle(mask_entries)
    mask = mask_entries.reshape((N, M))
    # We still have to make sure that there are no examples nor features with only missing values
    
    # One-dimensional numpy arrays with the number of missing values per example and number of missing values per feature.
    n_na_ex, n_na_feat = np.sum(a=mask, axis=1), np.sum(a=mask, axis=0)
    # One-dimensional numpy array containing the indexes of the examples which have only missing values.
    ind_mis_ex = np.nonzero(n_na_ex == M)[0]
    # One-dimensional numpy arrays indicating the examples having strictly less than M-1 missing values and the features having strictly less than N-1 missing values.
    av_ex, av_feat = (n_na_ex < M-1), (n_na_feat < N-1)
    # Managing the examples which have only missing values
    if ind_mis_ex.size > 0:
        # For each missing example
        for id_mis_ex in ind_mis_ex:
            # Randomly sampling a feature of the example indexed by id_mis_ex
            id_feat = rand_state.randint(low=0, high=M, size=1, dtype=np.int64)
            # Setting this feature as an observed data
            mask[id_mis_ex, id_feat] = False
            # One-dimensional numpy array containing the indexes of the examples having strictly less than M-1 missing values.
            ind_av_ex = np.nonzero(av_ex)[0]
            # Searching an example in which we can add a missing value
            while True:
                # Randomly sampling one of the available examples
                id_id_ex = rand_state.randint(low=0, high=ind_av_ex.size, size=1, dtype=np.int64)
                id_ex = ind_av_ex[id_id_ex]
                # Indexes of the features of the examples indexed by id_ex which are observed and available for introducing a missing data
                ind_av_feat_ex = np.nonzero(np.logical_and(np.logical_not(np.ravel(mask[id_ex,:])), av_feat))[0]
                # Stopping the loop if ind_av_feat_ex is not empty. Otherwise, removing id_ex from ind_av_ex.
                if ind_av_feat_ex.size > 0:
                    break
                else:
                    ind_av_ex = np.delete(arr=ind_av_ex, obj=id_id_ex)
            # Ramdomly sampling a feature from ind_av_feat_ex
            id_feat_ex = rand_state.choice(a=ind_av_feat_ex, size=1, replace=True, p=None)
            # Introducing a missing data in the feature id_feat_ex of the example id_ex
            mask[id_ex, id_feat_ex] = True
            # Updating n_na_ex
            n_na_ex[id_mis_ex] -= 1
            n_na_ex[id_ex] += 1
            # Updating n_na_feat
            n_na_feat[id_feat] -= 1
            n_na_feat[id_feat_ex] += 1
            # Updating av_ex. Notice that the number of missing values of the example indexed by id_mis_ex is M-1.
            av_ex[id_ex] = (n_na_ex[id_ex] < M-1)
            # Updating av_feat
            av_feat[id_feat] = (n_na_feat[id_feat] < N-1)
            av_feat[id_feat_ex] = (n_na_feat[id_feat_ex] < N-1)
    # One-dimensional numpy array containing the indexes of the features which have only missing values.
    ind_mis_feat = np.nonzero(n_na_feat == N)[0]
    # Managing the features which have only missing values
    if ind_mis_feat.size > 0:
        # For each missing feature
        for id_mis_feat in ind_mis_feat:
            # Randomly sampling an example of the feature indexed by id_mis_feat
            id_ex = rand_state.randint(low=0, high=N, size=1, dtype=np.int64)
            # Setting this example as an observed data
            mask[id_ex, id_mis_feat] = False
            # One-dimensional numpy array containing the indexes of the features having strictly less than N-1 missing values.
            ind_av_feat = np.nonzero(av_feat)[0]
            # Searching a feature in which we can add a missing value
            while True:
                # Randomly sampling one of the available features
                id_id_feat = rand_state.randint(low=0, high=ind_av_feat.size, size=1, dtype=np.int64)
                id_feat = ind_av_feat[id_id_feat]
                # Indexes of the examples of the feature indexed by id_feat which are observed and available for introducing a missing data
                ind_av_ex_feat = np.nonzero(np.logical_and(np.logical_not(np.ravel(mask[:,id_feat])), av_ex))[0]
                # Stopping the loop if ind_av_ex_feat is not empty. Otherwise, removing id_feat from ind_av_feat.
                if ind_av_ex_feat.size > 0:
                    break
                else:
                    ind_av_feat = np.delete(arr=ind_av_feat, obj=id_id_feat)
            # Ramdomly sampling an example from ind_av_ex_feat
            id_ex_feat = rand_state.choice(a=ind_av_ex_feat, size=1, replace=True, p=None)
            # Introducing a missing data in the feature id_feat_ex of the example id_ex
            mask[id_ex_feat, id_feat] = True
            # Updating n_na_ex
            n_na_ex[id_ex] -= 1
            n_na_ex[id_ex_feat] += 1
            # Updating n_na_feat
            n_na_feat[id_mis_feat] -= 1
            n_na_feat[id_feat] += 1
            # Updating av_ex.
            av_ex[id_ex] = (n_na_ex[id_ex] < M-1)
            av_ex[id_ex_feat] = (n_na_ex[id_ex_feat] < M-1)
            # Updating av_feat. Notice that the number of missing values of the feature indexed by id_mis_feat is N-1.
            av_feat[id_feat] = (n_na_feat[id_feat] < N-1)
    # Checking that there are no example with only missing data
    if np.any(a=(np.sum(a=mask, axis=1)==M)):
        raise ValueError("Error in function add_missing of module {module_name}: there are some examples with only missing data. Number of missing examples: {v}.".format(module_name=module_name, v=(np.sum(a=mask, axis=1)==M).sum()))
    # Checking that there are no feature with only missing data
    if np.any(a=(np.sum(a=mask, axis=0)==N)):
        raise ValueError("Error in function add_missing of module {module_name}: there are some features with only missing data. Number of missing features: {v}.".format(module_name=module_name, v=(np.sum(a=mask, axis=0)==N).sum()))
    # Checking that there is the correct number of observed data
    if np.logical_not(mask).sum() != n_obs:
        raise ValueError("Error in function add_missing of module {module_name}: there is not the correct number of observed data. Number of observed data = {v} while it should be {w}.".format(module_name=module_name, v=np.logical_not(mask).sum(), w=n_obs))
    # Checking that there is the correct number of missing data
    if mask.sum() != n_missing:
        raise ValueError("Error in function add_missing of module {module_name}: there is not the correct number of missing data. Number of missing data = {v} while it should be {w}.".format(module_name=module_name, v=mask.sum(), w=n_missing))
    # Defining the masked array and returning
    return np.ma.array(data=X, mask=mask, dtype=X.dtype, copy=True, subok=True, ndmin=0, fill_value=None, keep_mask=True, hard_mask=False, shrink=True, order=None)

##############################
############################## 
# Unsupervised DR quality assessment: rank-based criteria measuring the HD neighborhood preservation in the LD embedding [3, 4]. 
# These criteria are used in the experiments reported in [1]. 
# The main function is 'eval_dr_quality'. 
# See its documentation for details. It explains the meaning of the quality criteria and how to interpret them. 
# The demo at the end of this file presents how to use the 'eval_dr_quality' function. 
####################

def coranking(d_hd, d_ld):
    """
    Computation of the co-ranking matrix, as described in [4]. 
    The time complexity of this function is O(N**2 log(N)), where N is the number of data points.
    In:
    - d_hd: 2-D numpy array representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array representing the redundant matrix of pairwise distances in the LDS.
    Out:
    The (N-1)x(N-1) co-ranking matrix, where N = d_hd.shape[0].
    """
    # Computing the permutations to sort the rows of the distance matrices in HDS and LDS. 
    perm_hd = d_hd.argsort(axis=-1, kind='mergesort')
    perm_ld = d_ld.argsort(axis=-1, kind='mergesort')
    
    N = d_hd.shape[0]
    i = np.arange(N, dtype=np.int64)
    # Computing the ranks in the LDS
    R = np.empty(shape=(N,N), dtype=np.int64)
    for j in range(N):
        R[perm_ld[j,i],j] = i
    # Computing the co-ranking matrix
    Q = np.zeros(shape=(N,N), dtype=np.int64)
    for j in range(N):
        Q[i,R[perm_hd[j,i],j]] += 1
    # Returning
    return Q[1:,1:]

@numba.jit(nopython=True)
def eval_auc(arr):
    """
    Evaluates the AUC, as defined in [2].
    In:
    - arr: 1-D numpy array storing the values of a curve from K=1 to arr.size.
    Out:
    The AUC under arr, as defined in [2], with a log scale for K=1 to arr.size. 
    """
    i_all_k = 1.0/(np.arange(arr.size)+1.0)
    return np.float64(np.dot(arr, i_all_k))/(i_all_k.sum())

@numba.jit(nopython=True)
def eval_rnx(Q):
    """
    Evaluate R_NX(K) for K = 1 to N-2, as defined in [5]. N is the number of data points in the data set.
    The time complexity of this function is O(N^2).
    In:
    - Q: a 2-D numpy array representing the (N-1)x(N-1) co-ranking matrix of the embedding. 
    Out:
    A 1-D numpy array with N-2 elements. Element i contains R_NX(i+1).
    """
    N_1 = Q.shape[0]
    N = N_1 + 1
    # Computing Q_NX
    qnxk = np.empty(shape=N_1, dtype=np.float64)
    acc_q = 0.0
    for K in range(N_1):
        acc_q += (Q[K,K] + np.sum(Q[K,:K]) + np.sum(Q[:K,K]))
        qnxk[K] = acc_q/((K+1)*N)
    # Computing R_NX
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    rnxk = (N_1*qnxk[:N_1-1]-arr_K)/(N_1-arr_K)
    # Returning
    return rnxk

def eval_dr_quality(d_hd, d_ld):
    """
    Compute the DR quality assessment criteria R_{NX}(K) and AUC, as defined in [2, 3, 4, 5] and as employed in the experiments reported in [1].
    These criteria measure the neighborhood preservation around the data points from the HDS to the LDS. 
    Based on the HD and LD distances, the sets v_i^K (resp. n_i^K) of the K nearest neighbors of data point i in the HDS (resp. LDS) can first be computed. 
    Their average normalized agreement develops as Q_{NX}(K) = (1/N) * \sum_{i=1}^{N} |v_i^K \cap n_i^K|/K, where N refers to the number of data points and \cap to the set intersection operator. 
    Q_{NX}(K) ranges between 0 and 1; the closer to 1, the better.
    As the expectation of Q_{NX}(K) with random LD coordinates is equal to K/(N-1), which is increasing with K, R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K) enables more easily comparing different neighborhood sizes K. 
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1. 
    The R_{NX}(K) values for K=1 to N-2 can be displayed as a curve with a log scale for K, as closer neighbors typically prevail. 
    The area under the resulting curve (AUC) is a scalar score which grows with DR quality, quantified at all scales with an emphasis on small ones.
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random. 
    In: 
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    Out: a tuple with
    - a 1-D numpy array with N-2 elements. Element i contains R_{NX}(i+1).
    - the AUC of the R_{NX}(K) curve with a log scale for K, as defined in [2].
    Remark:
    - The time complexity to evaluate the quality criteria is O(N**2 log(N)). It is the time complexity to compute the co-ranking matrix. R_{NX}(K) can then be evaluated for all K=1, ..., N-2 in O(N**2). 
    """
    # Computing the co-ranking matrix of the embedding, and the R_{NX}(K) curve.
    rnxk = eval_rnx(Q=coranking(d_hd=d_hd, d_ld=d_ld))
    # Computing the AUC, and returning.
    return rnxk, eval_auc(rnxk)

##############################
############################## 
# Supervised DR quality assessment: accuracy of a KNN classifier in the LD embedding [6]. 
# See the documentation of the 'knngain' function for details. It explains the meaning of the supervised quality criteria and how to interpret them. 
####################

@numba.jit(nopython=True)
def knngain(d_hd, d_ld, labels):
    """
    Compute the KNN gain curve and its AUC, as defined in [6]. 
    If c_i refers to the class label of data point i, v_i^K (resp. n_i^K) to the set of the K nearest neighbors of data point i in the HDS (resp. LDS), and N to the number of data points, the KNN gain develops as G_{NN}(K) = (1/N) * \sum_{i=1}^{N} (|{j \in n_i^K such that c_i=c_j}|-|{j \in v_i^K such that c_i=c_j}|)/K.
    It averages the gain (or loss, if negative) of neighbors of the same class around each point, after DR. 
    Hence, a positive value correlates with likely improved KNN classification performances.
    As the R_{NX}(K) curve from the unsupervised DR quality assessment, the KNN gain G_{NN}(K) can be displayed with respect to K, with a log scale for K. 
    A global score summarizing the resulting curve is provided by its area (AUC). 
    In: 
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    - labels: 1-D numpy array with N elements, containing integers indicating the class labels of the data points. 
    Out: 
    A tuple with:
    - a 1-D numpy array of floats with N-1 elements, storing the KNN gain for K=1 to N-1. 
    - the AUC of the KNN gain curve, with a log scale for K.
    """
    # Number of data points
    N = d_hd.shape[0]
    N_1 = N-1
    k_hd = np.zeros(shape=N_1, dtype=np.int64)
    k_ld = np.zeros(shape=N_1, dtype=np.int64)
    # For each data point
    for i in range(N):
        c_i = labels[i]
        di_hd = d_hd[i,:].argsort(kind='mergesort')
        di_ld = d_ld[i,:].argsort(kind='mergesort')
        # Making sure that i is first in di_hd and di_ld
        for arr in [di_hd, di_ld]:
            for idj, j in enumerate(arr):
                if j == i:
                    idi = idj
                    break
            if idi != 0:
                arr[idi] = arr[0]
            arr = arr[1:]
        for k in range(N_1):
            if c_i == labels[di_hd[k]]:
                k_hd[k] += 1
            if c_i == labels[di_ld[k]]:
                k_ld[k] += 1
    # Computing the KNN gain
    gn = (k_ld.cumsum() - k_hd.cumsum()).astype(np.float64)/((1.0+np.arange(N_1))*N)
    # Returning the KNN gain and its AUC
    return gn, eval_auc(gn)

##############################
############################## 
# Plot functions reproducing some of the figures in [1]. 
# The main functions are 'viz_2d_emb' and 'viz_qa'.
# Their documentations detail their parameters. 
# The demo at the end of this file presents how to use these functions. 
####################

def rstr(v, d=2):
    """
    Rounds v with d digits and returns it as a string. If it starts with 0, it is omitted. 
    In:
    - v: a number. 
    - d: number of digits to keep.
    Out:
    A string representing v rounded with d digits. If it starts with 0, it is omitted. 
    """
    p = 10.0**d
    v = str(int(round(v*p))/p)
    if v[0] == '0':
        v = v[1:]
    elif (len(v) > 3) and (v[:3] == '-0.'):
        v = "-.{a}".format(a=v[3:])
    return v

def check_create_dir(path):
    """
    Create a directory at the specified path only if it does not already exist.
    """
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def save_show_fig(fname=None, f_format=None, dpi=300):
    """
    Save or show a figure.
    In:
    - fname: filename to save the figure, without the file extension. If None, the figure is shown.
    - f_format: format to save the figure. If None, set to pdf. 
    - dpi: DPI to save the figure.
    Out: 
    A figure is shown if fname is None, and saved otherwise.
    """
    if fname is None:
        plt.show()
    else:
        if f_format is None:
            f_format = 'pdf'
        # Checking whether a folder needs to be created
        check_create_dir(fname)
        # Saving the figure
        plt.savefig("{fname}.{f_format}".format(fname=fname, f_format=f_format), format=f_format, dpi=dpi, bbox_inches='tight', facecolor='w', edgecolor='w', orientation='portrait', papertype='a4', transparent=False, pad_inches=0.1, frameon=None)

def viz_2d_emb(X, vcol, tit='', fname=None, f_format=None, cmap='rainbow', sdot=20, marker='o', a_scat=0.8, edcol_scat='face', stit=15, lw=2.0):
    """
    Plot a 2-D embedding of a data set.
    In:
    - X: a 2-D numpy array with shape (N, 2), where N is the number of data points to represent in the 2-D embedding.
    - vcol: a 1-D numpy array with N elements, indicating the colors of the data points in the colormap.
    - tit: title of the figure.
    - fname, f_format: path. Same as in save_show_fig.
    - cmap: colormap.
    - sdot: size of the dots.
    - marker: marker.
    - a_scat: alpha used to plot the data points.
    - edcol_scat: edge color for the points of the scatter plot. From the official documentation: "If None, defaults to (patch.edgecolor). If 'face', the edge color will always be the same as the face color. If it is 'none', the patch boundary will not be drawn. For non-filled markers, the edgecolors kwarg is ignored; color is determined by c.".
    - stit: fontsize of the title of the figure.
    - lw: linewidth for the scatter plot.
    Out:
    Same as save_show_fig.
    """  
    global module_name
    
    # Checking X
    if X.ndim != 2:
        raise ValueError("Error in function viz_2d_emb of {module_name}: X must be a numpy array with shape (N, 2), where N is the number of data points to plot in the 2-D embedding.".format(module_name=module_name))
    if X.shape[1] != 2:
        raise ValueError("Error in function viz_2d_emb of {module_name}: X must have 2 columns.".format(module_name=module_name))
    
    # Computing the limits of the axes
    xmin = X[:,0].min()
    xmax = X[:,0].max()
    ev = (xmax-xmin)*0.05
    x_lim = np.asarray([xmin-ev, xmax+ev])
    
    ymin = X[:,1].min()
    ymax = X[:,1].max()
    ev = (ymax-ymin)*0.05
    y_lim = np.asarray([ymin-ev, ymax+ev])
    
    vmc = vcol.min()
    vMc = vcol.max()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Setting the limits of the axes
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    # Plotting the data points
    ax.scatter(X[:,0], X[:,1], c=vcol, cmap=cmap, s=sdot, marker=marker, alpha=a_scat, edgecolors=edcol_scat, vmin=vmc, vmax=vMc, linewidths=lw)
    
    # Removing the ticks
    ax.set_xticks([], minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([], minor=False)
    ax.set_yticks([], minor=False)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels([], minor=False)
    
    ax.set_title(tit, fontsize=stit)
    plt.tight_layout()
    
    # Saving or showing the figure, and closing
    save_show_fig(fname=fname, f_format=f_format)
    plt.close()

def viz_qa(Ly, fname=None, f_format=None, ymin=None, ymax=None, Lmarkers=None, Lcols=None, Lleg=None, Lls=None, Lmedw=None, Lsdots=None, lw=2, markevery=0.1, tit='', xlabel='', ylabel='', alpha_plot=0.9, alpha_leg=0.8, stit=25, sax=20, sleg=15, zleg=1, loc_leg='best', ncol_leg=1, lMticks=10, lmticks=5, wMticks=2, wmticks=1, nyMticks=11, mymticks=4, grid=True, grid_ls='solid', grid_col='lightgrey', grid_alpha=0.7, xlog=True):
    """
    Plot the DR quality criteria curves. 
    In: 
    - Ly: list of 1-D numpy arrays. The i^th array gathers the y-axis values of a curve from x=1 to x=Ly[i].size, with steps of 1. 
    - fname, f_format: path. Same as in save_show_fig.
    - ymin, ymax: minimum and maximum values of the y-axis. If None, ymin (resp. ymax) is set to the smallest (resp. greatest) value among [y.min() for y in Ly] (resp. [y.max() for y in Ly]).
    - Lmarkers: list with the markers for each curve. If None, some pre-defined markers are used.
    - Lcols: list with the colors of the curves. If None, some pre-defined colors are used.
    - Lleg: list of strings, containing the legend entries for each curve. If None, no legend is shown.
    - Lls: list of the linestyles ('solid', 'dashed', ...) of the curves. If None, 'solid' style is employed for all curves. 
    - Lmedw: list with the markeredgewidths of the curves. If None, some pre-defined value is employed. 
    - Lsdots: list with the sizes of the markers. If None, some pre-defined value is employed.
    - lw: linewidth for all the curves. 
    - markevery: approximately 1/markevery markers are displayed for each curve. Set to None to mark every dot.
    - tit: title of the plot.
    - xlabel, ylabel: labels for the x- and y-axes.
    - alpha_plot: alpha for the curves.
    - alpha_leg: alpha for the legend.
    - stit: fontsize for the title.
    - sax: fontsize for the labels of the axes. 
    - sleg: fontsize for the legend.
    - zleg: zorder for the legend. Set to 1 to plot the legend behind the data, and to None to keep the default value.
    - loc_leg: location of the legend ('best', 'upper left', ...).
    - ncol_leg: number of columns to use in the legend.
    - lMticks: length of the major ticks on the axes.
    - lmticks: length of the minor ticks on the axes.
    - wMticks: width of the major ticks on the axes.
    - wmticks: width of the minor ticks on the axes.
    - nyMticks: number of major ticks on the y-axis (counting ymin and ymax).
    - mymticks: there are 1+mymticks*(nyMticks-1) minor ticks on the y axis.
    - grid: True to add a grid, False otherwise.
    - grid_ls: linestyle of the grid.
    - grid_col: color of the grid.
    - grid_alpha: alpha of the grid.
    - xlog: True to produce a semilogx plot and False to produce a plot. 
    Out:
    A figure is shown. 
    """
    # Number of curves
    nc = len(Ly)
    # Checking the parameters
    if ymin is None:
        ymin = np.min(np.asarray([arr.min() for arr in Ly]))
    if ymax is None:
        ymax = np.max(np.asarray([arr.max() for arr in Ly]))
    if Lmarkers is None:
        Lmarkers = ['x']*nc
    if Lcols is None:
        Lcols = ['blue']*nc
    if Lleg is None:
        Lleg = [None]*nc
        add_leg = False
    else:
        add_leg = True
    if Lls is None:
        Lls = ['solid']*nc
    if Lmedw is None:
        Lmedw = [float(lw)/2.0]*nc
    if Lsdots is None:
        Lsdots = [12]*nc
    
    # Setting the limits of the y-axis
    y_lim = [ymin, ymax]
    
    # Defining the ticks on the y-axis
    yMticks = np.linspace(start=ymin, stop=ymax, num=nyMticks, endpoint=True, retstep=False)
    ymticks = np.linspace(start=ymin, stop=ymax, num=1+mymticks*(nyMticks-1), endpoint=True, retstep=False)
    yMticksLab = [rstr(v) for v in yMticks]
    
    # Initial values for xmin and xmax
    xmin, xmax = 1, -np.inf
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if xlog:
        fplot = ax.semilogx
    else:
        fplot = ax.plot
    
    # Plotting the data
    for id, y in enumerate(Ly):
        x = np.arange(start=1, step=1, stop=y.size+0.5, dtype=np.int64)
        xmax = max(xmax, x[-1])
        fplot(x, y, label=Lleg[id], alpha=alpha_plot, color=Lcols[id], linestyle=Lls[id], lw=lw, marker=Lmarkers[id], markeredgecolor=Lcols[id], markeredgewidth=Lmedw[id], markersize=Lsdots[id], dash_capstyle='round', solid_capstyle='round', dash_joinstyle='round', solid_joinstyle='round', markerfacecolor=Lcols[id], markevery=markevery)
    
    # Setting the limits of the axes
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(y_lim)
    
    # Setting the major and minor ticks on the y-axis 
    ax.set_yticks(yMticks, minor=False)
    ax.set_yticks(ymticks, minor=True)
    ax.set_yticklabels(yMticksLab, minor=False, fontsize=sax)
    
    # Defining the legend
    if add_leg:
        leg = ax.legend(loc=loc_leg, fontsize=sleg, markerfirst=True, fancybox=True, framealpha=alpha_leg, ncol=ncol_leg)
        if zleg is not None:
            leg.set_zorder(zleg)
    
    # Setting the size of the ticks labels on the x axis
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(sax)
    
    # Setting ticks length and width
    ax.tick_params(axis='both', length=lMticks, width=wMticks, which='major')
    ax.tick_params(axis='both', length=lmticks, width=wmticks, which='minor')
    
    # Setting the positions of the labels
    ax.xaxis.set_tick_params(labelright=False, labelleft=True)
    ax.yaxis.set_tick_params(labelright=False, labelleft=True)
    
    # Adding the grids
    if grid:
        ax.xaxis.grid(True, linestyle=grid_ls, which='major', color=grid_col, alpha=grid_alpha)
        ax.yaxis.grid(True, linestyle=grid_ls, which='major', color=grid_col, alpha=grid_alpha)
    ax.set_axisbelow(True)
    
    ax.set_title(tit, fontsize=stit)
    ax.set_xlabel(xlabel, fontsize=sax)
    ax.set_ylabel(ylabel, fontsize=sax)
    plt.tight_layout()
    
    # Saving or showing the figure, and closing
    save_show_fig(fname=fname, f_format=f_format)
    plt.close()

##############################
############################## 
# Demo presenting how to use the main functions of this file.
####################

def load_yacht_hydro():
    """
    Load the Yacht Hydrodynamics data set from the UCI Machine Learning Repository. 
    It is available at: http://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics (last consulted on Dec 30, 2019). 
    The targets of the examples are not loaded. 
    Out:
    A tuple with:
    - X: a 2-D numpy array with shape (308, 6) with one example per row and one feature per column. Z-score standardization is applied on the features.
    - labels: a 1-D numpy array with shape (308,) such that element i is the label of example i. It corresponds to the index of the hull form related to example i. 
    """
    # Number of different hull forms
    n_hull = 22
    
    # Loading the data
    data = pandas.read_csv('./data/yacht_hydro.txt', names=['a', 'b', 'c', 'd', 'e', 'f', 'g'], sep=',')
    
    # Removing the targets
    del data['g']
    
    # Converting the data frame into a 2-D numpy array
    X = data.values.astype(dtype=np.float64)
    
    # Number of samples
    N = X.shape[0]
    # Number of samples per hull form
    n_samp_hull = int(round(N/n_hull))
    
    # Defining one label per hull form
    labels = np.zeros(shape=N, dtype=np.int64)
    for i in range(n_hull):
        j = i*n_samp_hull
        labels[j:j+n_samp_hull] = i
    labels = labels.astype(np.float64)
    
    # Permutation of the examples
    ind_ex = np.arange(start=0, step=1, stop=N, dtype=np.int64)
    np.random.RandomState(1).shuffle(ind_ex)
    X = X[ind_ex,:]
    labels = labels[ind_ex]
    
    # Z-score standardization
    X = scipy.stats.mstats.zscore(a=X, axis=0, ddof=1.0)
    
    return X, labels

if __name__ == '__main__':
    print("===================================================")
    print("===== Starting the demo of dr_missing_data.py =====")
    print("===================================================")
    
    print(' !!!                                                                                                              !!! ')
    print(" !!! Do not forget to download the 'data' folder at the same path as this file, otherwise the demo will not work. !!! ")
    print(' !!!                                                                                                              !!! ')
    
    ###
    ###
    ###
    print('- Loading the Yacht Hydrodynamics data set, from the UCI Machine Learning Repository')
    # TIP: to change the employed data set, you just need to modify the load_yacht_hydro function to provide different values for X_complete and labels.
    X_complete, labels = load_yacht_hydro()
    print('===')
    print('===')
    print('===')
    
    ###
    ###
    ###
    # Function to compute a 2-D numpy array containing the pairwise distances in a complete data set X. This function is used to compute the HD distances used both in multi-scale SNE and for the DR quality assessment.
    compute_dist_HD = eucl_dist_matr
    # Function to compute a 2-D numpy array containing the pairwise distances in a complete data set X. This function is used to compute the LD distances for the DR quality assessment. Note that in multi-scale SNE, the LD embedding is computed using Euclidean distances in the LD space, independently of the value of compute_dist_LD_qa.
    compute_dist_LD_qa = eucl_dist_matr
    # Lists to provide as parameters to viz_qa, to visualize the DR quality assessment as conducted in [1].
    L_rnx, Lmarkers, Lcols, Lleg_rnx, Lls, Lmedw, Lsdots = [], [], [], [], [], [], []
    
    ###
    ###
    ###
    print('- Computing the pairwise Euclidean distances in the complete HD data set')
    t0 = time.time()
    dm_hd_complete = compute_dist_HD(X_complete)
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    
    ###
    ###
    ###
    # Targeted dimension of the LD embeddings
    dim_LDS = 2
    # Initialization for multi-scale SNE. Check the 'init' parameter of mssne_implem for a description. In particular, init_mssne can be equal to 'pca', 'random' or a 2-D numpy array containing the initial LD coordinates of the data points. 
    init_mssne = 'random'
    # dm_fct_mssne parameter of function mssne_na_mmg. It is a function used to compute the pairwise HD distances (NOT squared) between the data points in the imputed data sets provided to multi-scale SNE when reducing the dimension of an incomplete data set. If dm_fct_mssne is None, Euclidean distance is used in the HD space in multi-scale SNE by default. An example of a valid function for the dm_fct_mssne parameter is the eucl_dist_matr one. This function is set to compute_dist_HD for consistency: the same HD distance is used by multi-scale SNE in both complete and incomplete data cases, and also for the DR quality assessment. It is set to None if compute_dist_HD is set to eucl_dist_matr, as setting dm_fct_mssne to None reduces to using Euclidean HD distances. If you change dm_fct_mssne, make sure to change compute_dist_HD accordingly to use the same HD distances in multi-scale SNE in both complete and incomplete data cases. 
    dm_fct_mssne = None if (compute_dist_HD == eucl_dist_matr) else compute_dist_HD
    # fitU_mssne parameter of function mssne_na_mmg
    fitU_mssne = False
    # Random seed for multi-scale SNE. This ensures that the same LD initialization is used for all applications of multi-scale SNE.
    seed_mssne = 40
    
    ###
    ###
    ###
    print('- Applying multi-scale SNE on the complete data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
    t0 = time.time()
    # TIP: you could set dm_hds to None in the following line if you do not want to precompute the pairwise HD distances dm_hd_complete; in this case, pairwise HD Euclidean distances would then be computed based on X_complete in mssne_implem. You can also use other HD distances than the Euclidean one for the dm_hds parameter: you just need to modify the above compute_dist_HD function to compute the 2-D numpy array storing the pairwise distances of your choice. Note that you can provide the LD coordinates to use for the initialization of multi-scale SNE by setting init_mssne to a 2-D numpy.ndarray containing the initial LD positions, with one example per row and one LD dimension per column, init_mssne[i,:] containing the initial LD coordinates related to the HD sample X_complete[i,:].
    X_ld_completeHD, sigma_ij_completeHD = mssne_implem(X_hds=X_complete, init=init_mssne, n_components=dim_LDS, ret_sim_hds=True, fit_U=fitU_mssne, dm_hds=dm_hd_complete, seed_mssne=seed_mssne)
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    
    ###
    ###
    ###
    print('- Evaluating the DR quality of the LD embedding obtained based on the complete data set')
    t0 = time.time()
    rnx_complete, auc_complete = eval_dr_quality(d_hd=dm_hd_complete, d_ld=compute_dist_LD_qa(X_ld_completeHD))
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    print('AUC: {v}'.format(v=rstr(auc_complete, 4)))
    
    # Updating the lists for viz_qa
    L_rnx.append(rnx_complete)
    Lmarkers.append('>')
    Lcols.append('#FF8000')
    Lleg_rnx.append('Complete data')
    Lls.append('solid')
    Lmedw.append(0.5)
    Lsdots.append(10)
    
    ###
    ###
    ###
    print('- Plotting the LD embedding obtained based on the complete data set')
    print('If a figure is shown, close it to continue.')
    # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information. 
    viz_2d_emb(X=X_ld_completeHD, vcol=labels, tit='LD embedding for complete data', fname=None, f_format=None)
    print('===')
    print('===')
    print('===')
    
    ###
    ###
    ###
    # Proportion of missing values to introduce in X_complete
    p_na = 0.05
    # Random state to introduce missing values in X_complete
    rand_state = np.random.RandomState(7)
    # Function to compute the Q_{KL} score as defined in [1]
    q_KL = lambda sim: scipy.special.rel_entr(sigma_ij_completeHD, sim).sum()
    
    ###
    ###
    ###
    print('- Introducing {v}% missing values in the data set'.format(v=rstr(100*p_na)))
    # The following loop is to make sure that the incomplete data set does not have duplicated samples. 
    go = True
    while go:
        X_missing = add_missing(X=X_complete, p=p_na, rand_state=rand_state)
        # Gathering the mask of X_missing. A True entry indicates a missing data. 
        X_mask = X_missing.mask
        # Filling the missing entries with zeros
        X = X_missing.filled(fill_value=0.0)
        # Checking whether there are two identical samples in X. If yes, then looping.
        go = contains_ident_ex(X)
    print('===')
    print('===')
    print('===')
    
    ###
    ###
    ###
    # In the following loop, the methodology detailed in [1] to deal with missing data is applied. When fit_K is True, a Gaussian mixture is learned on the incomplete data set, and its number of components is tuned as in [1]. When fit_K is False, a single multivariate Gaussian is learned on the incomplete data set. This reproduces the same experimental setting as in [1], in which the performances using a Gaussian mixture and a single multivariate Gaussian are compared. 
    for id_loop, fit_K in enumerate([True, False]):
        str_data_model = 'Gaussian mixture' if fit_K else 'single multivariate Gaussian'
        
        ###
        ###
        ###
        print('- Reducing the dimension of the incomplete HD data set using multi-scale SNE by applying the methodology presented in (de Bodt et al, IEEE TNNLS, 2019) [1] to manage the missing data. (#{id_loop}/2)'.format(id_loop=id_loop+1))
        print('---> A {str_data_model} is first learned on the incomplete database.'.format(str_data_model=str_data_model))
        print('---> Then, the LD embeddings resulting from both the multiple imputations and the conditional mean imputation methods are computed, as described in [1].')
        t0 = time.time()
        # TIP: The dm_fct_mssne parameter determines the distance employed in the HD space by multi-scale SNE. Check the documentation of the mssne_na_mmg function for more information.
        ret_mi, ret_si, vg = mssne_na_mmg(X=X, X_mask=X_mask, dr_mi=True, dr_si=True, fit_K=fit_K, dim_LDS=dim_LDS, init_mssne=init_mssne, dm_fct_mssne=dm_fct_mssne, fitU_mssne=fitU_mssne, seed_mmg=3, seed_mssne=seed_mssne)
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=rstr(t)))
        X_lds_mi, sim_hd_na_mi, t_na_mi, t_dr_mi, X_mi = ret_mi
        X_lds_si, sim_hd_na_si, t_na, t_dr_si = ret_si
        print('Time to fit the {str_data_model} model: {t} seconds.'.format(str_data_model=str_data_model, t=rstr(t_na)))
        if fit_K:
            n_gaussians = vg[0][4].size
            print('Tuned number of Gaussian components: {n_gaussians}.'.format(n_gaussians=n_gaussians))
        print('Time to fit the {str_data_model} model and perform the multiple imputations: {t} seconds.'.format(str_data_model=str_data_model, t=rstr(t_na_mi)))
        print('Time to minimize the expected cost function of multi-scale SNE, which is estimated using multiple imputations: {t} seconds.'.format(t=rstr(t_dr_mi)))
        print('Time to apply multi-scale SNE after filling the missing data thanks to conditional mean imputation: {t} seconds.'.format(t=rstr(t_dr_si)))
        print('Q_KL score (as defined in [1]) when using multiple imputations: {v}'.format(v=rstr(q_KL(sim_hd_na_mi), 4)))
        print('Q_KL score (as defined in [1]) when using conditional mean imputation: {v}'.format(v=rstr(q_KL(sim_hd_na_si), 4)))
        
        ###
        ###
        ###
        print('- Evaluating the DR quality of the LD embedding obtained for the incomplete data set thanks to multiple imputations')
        t0 = time.time()
        rnx_mi, auc_mi = eval_dr_quality(d_hd=dm_hd_complete, d_ld=compute_dist_LD_qa(X_lds_mi))
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=rstr(t)))
        print('AUC: {v}'.format(v=rstr(auc_mi, 4)))
        
        # Updating the lists for viz_qa
        L_rnx.append(rnx_mi)
        if fit_K:
            Lmarkers.append('$\star$')
            Lcols.append('red')
            Lleg_rnx.append('MIMG')
            Lls.append('solid')
            Lmedw.append(1.2)
            Lsdots.append(10)
        else:
            Lmarkers.append('s')
            Lcols.append('green')
            Lleg_rnx.append('MIG')
            Lls.append('solid')
            Lmedw.append(0.5)
            Lsdots.append(10)
        
        print('- Evaluating the DR quality of the LD embedding obtained for the incomplete data set thanks to conditional mean imputation')
        t0 = time.time()
        rnx_si, auc_si = eval_dr_quality(d_hd=dm_hd_complete, d_ld=compute_dist_LD_qa(X_lds_si))
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=rstr(t)))
        print('AUC: {v}'.format(v=rstr(auc_si, 4)))
        
        # Updating the lists for viz_qa
        L_rnx.append(rnx_si)
        if fit_K:
            Lmarkers.append('x')
            Lcols.append('blue')
            Lleg_rnx.append('IMG')
            Lls.append('solid')
            Lmedw.append(1.5)
            Lsdots.append(10)
        else:
            Lmarkers.append('^')
            Lcols.append('magenta')
            Lleg_rnx.append('IG')
            Lls.append('solid')
            Lmedw.append(0.5)
            Lsdots.append(10)
        
        ###
        ###
        ###
        if fit_K:
            print('- Plotting the LD embedding obtained by multi-scale SNE for the incomplete data set thanks to multiple imputations using a Gaussian mixture model')
            print('If a figure is shown, close it to continue.')
            # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information. 
            viz_2d_emb(X=X_lds_mi, vcol=labels, tit='LD embedding for incomplete data (MIMG)', fname=None, f_format=None)
        print('===')
        print('===')
        print('===')
    
    ###
    ###
    ###
    print('- Reducing the dimension of the incomplete HD data set using multi-scale SNE by applying the ICkNNI method to manage the missing data, as reported in the experiments of (de Bodt et al, IEEE TNNLS, 2019) [1].')
    t0 = time.time()
    # TIP: The dm_fct_mssne parameter determines the distance employed in the HD space by multi-scale SNE. Check the documentation of the mssne_na_icknni function for more information.
    # TIP: Check the 'dnn_fct' parameter of mssne_na_icknni to use other distances than the Euclidean one in ICkNNI.
    X_ld_inn, sim_hd_na_inn, t_na_inn, t_dr_inn = mssne_na_icknni(X=X, X_mask=X_mask, k_nn=5, dim_LDS=dim_LDS, init_mssne=init_mssne, dm_fct_mssne=dm_fct_mssne, fitU_mssne=fitU_mssne, seed_mssne=seed_mssne)
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    print('Time to apply ICkNNI: {t} seconds.'.format(t=rstr(t_na_inn)))
    print('Time to apply multi-scale SNE after filling the missing data thanks to ICkNNI: {t} seconds.'.format(t=rstr(t_dr_inn)))
    print('Q_KL score (as defined in [1]) when using ICkNNI: {v}'.format(v=rstr(q_KL(sim_hd_na_inn), 4)))
    
    ###
    ###
    ###
    print('- Evaluating the DR quality of the LD embedding obtained for the incomplete data set thanks to ICkNNI')
    t0 = time.time()
    rnx_inn, auc_inn = eval_dr_quality(d_hd=dm_hd_complete, d_ld=compute_dist_LD_qa(X_ld_inn))
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    print('AUC: {v}'.format(v=rstr(auc_inn, 4)))
    print('===')
    print('===')
    print('===')
    
    # Updating the lists for viz_qa
    L_rnx.append(rnx_inn)
    Lmarkers.append('|')
    Lcols.append('#00CCCC')
    Lleg_rnx.append('INN')
    Lls.append('solid')
    Lmedw.append(1.5)
    Lsdots.append(10)
    
    ###
    ###
    ###
    print('- Reducing the dimension of the incomplete HD data set using multi-scale SNE by applying the mean imputation method to manage the missing data, as reported in the experiments of (de Bodt et al, IEEE TNNLS, 2019) [1].')
    t0 = time.time()
    # TIP: The dm_fct_mssne parameter determines the distance employed in the HD space by multi-scale SNE. Check the documentation of the mssne_na_mu_si function for more information.
    X_ld_ime, sim_hd_na_ime, t_na_ime, t_dr_ime = mssne_na_mu_si(X=X, X_mask=X_mask, dim_LDS=dim_LDS, init_mssne=init_mssne, dm_fct_mssne=dm_fct_mssne, fitU_mssne=fitU_mssne, seed_mssne=seed_mssne)
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    print('Time to apply the mean imputation method: {t} seconds.'.format(t=rstr(t_na_ime)))
    print('Time to apply multi-scale SNE after filling the missing data thanks to mean imputation: {t} seconds.'.format(t=rstr(t_dr_ime)))
    print('Q_KL score (as defined in [1]) when using mean imputation: {v}'.format(v=rstr(q_KL(sim_hd_na_ime), 4)))
    
    ###
    ###
    ###
    print('- Evaluating the DR quality of the LD embedding obtained for the incomplete data set thanks to mean imputation')
    t0 = time.time()
    rnx_ime, auc_ime = eval_dr_quality(d_hd=dm_hd_complete, d_ld=compute_dist_LD_qa(X_ld_ime))
    t = time.time() - t0
    print('Done. It took {t} seconds.'.format(t=rstr(t)))
    print('AUC: {v}'.format(v=rstr(auc_ime, 4)))
    print('===')
    print('===')
    print('===')
    
    # Updating the lists for viz_qa
    L_rnx.append(rnx_ime)
    Lmarkers.append('o')
    Lcols.append('black')
    Lleg_rnx.append('IMe')
    Lls.append('solid')
    Lmedw.append(0.5)
    Lsdots.append(10)
    
    ###
    ###
    ###
    print('- Plotting the results of the DR quality assessment, as reported in (de Bodt et al, IEEE TNNLS, 2019) [1].')
    print('---> Remark: the same legend entries are employed as in the experiments of [1].')
    print('If a figure is shown, close it to continue.')
    # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information. 
    viz_qa(Ly=L_rnx, Lmarkers=Lmarkers, Lcols=Lcols, Lleg=Lleg_rnx, Lls=Lls, Lmedw=Lmedw, Lsdots=Lsdots, tit='DR quality', xlabel='Neighborhood size $K$', ylabel='$R_{NX}(K)$', fname=None, f_format=None, ncol_leg=2)
    print('===')
    print('===')
    print('===')
    
    ###
    ###
    ###
    print('*********************')
    print('***** Done! :-) *****')
    print('*********************')






