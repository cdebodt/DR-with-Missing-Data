# DR-with-Missing-Data
Nonlinear Dimensionality Reduction with Missing Data using Parametric Multiple Imputations

==========


%%%% !!! IMPORTANT NOTE !!! %%%%
At the end of the dr_missing_data.py file, a demo presents how this python code can be used. Running this file (python dr_missing_data.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. Do not forget to download the 'data' folder next to the dr_missing_data.py file to be able to run the demo. 
%%%% !!!                !!! %%%%

The python code dr_missing_data.py implements a framework to deal with missing data in dimensionality reduction (DR). 
The methodology which is implemented is described in the article "Nonlinear Dimensionality Reduction with Missing Data using Parametric Multiple Imputations", from Cyril de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, published in IEEE Transactions on Neural Networks and Learning Systems, in 2019. 
Link to retrieve the article: https://ieeexplore.ieee.org/abstract/document/8447227
At the end of the dr_missing_data.py file, a demo presents how this python code can be used. Running this file (python dr_missing_data.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. The tested versions of the imported packages are specified below. 
Do not forget to download the 'data' folder next to the dr_missing_data.py file to be able to run the demo. 

If you use the dr_missing_data.py code or the article, please cite as: 
- de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2019). Nonlinear Dimensionality Reduction With Missing Data Using Parametric Multiple Imputations. IEEE transactions on neural networks and learning systems, 30(4), 1166-1179.
- BibTeX entry:
@article{cdb2019drnap,
  title={{N}onlinear {D}imensionality {R}eduction with {M}issing {D}ata using {P}arametric {M}ultiple {I}mputations},
  author={de Bodt, C. and Mulders, D. and Verleysen, M. and Lee, J. A.},
  journal={{IEEE} Trans. Neural Netw. Learn. Syst.},
  volume={30},
  number={4},
  pages={1166--1179},
  year={2019}
}

The main functions of the dr_missing_data.py file are:
- 'mssne_implem': nonlinear dimensionality reduction through multi-scale SNE (Ms SNE), as presented in the reference [2] below and summarized in [1]. This function enables reducing the dimension of a complete data set. 
- 'gmm_fit_K', 'gmm_sev_em_fitting' and 'gmm_sev_sampling': Gaussian mixture modeling of a complete or incomplete data set, as presented in [7, 8, 9] and summarized in [1]. These functions respectively enable to:
---> 'gmm_fit_K': fit a Gaussian mixture model on a complete or incomplete data set. The number of mixture components is automatically determined and tuned as detailed in [1]. 
---> 'gmm_sev_em_fitting': fit a Gaussian mixture model with K components on a complete or incomplete data set. The number K of components is a parameter of the function. Setting it to 1 fits a single multivariate Gaussian on the data set, while setting K to 2 fits two Gaussian components on the data set, etc. 
---> 'gmm_sev_sampling': draw samples from a Gaussian mixture model. 
- 'icknni_implem': implementation of the ICkNNI method as proposed in [15] and employed in the experiments of [1] for the comparison of the performances of the methods. This function enables performing a single imputation of the missing entries in a data set. 
- 'mssne_na_mmg': nonlinear dimensionality reduction through multi-scale SNE of an incomplete data set, using the methodology presented in [1]. This function enables applying multi-scale SNE on a database with missing values by first fitting a Gaussian mixture model on the data set and then dealing with the missing entries either thanks to multiple imputations or conditional mean imputation. 
- 'eval_dr_quality': unsupervised evaluation of the quality of a low-dimensional embedding, as introduced in [3, 4] and employed and summarized in [1, 2, 5]. This function enables computing quality assessment criteria measuring the neighborhood preservation from the high-dimensional space to the low-dimensional one. The documentation of the function explains the meaning of the criteria and how to interpret them.
- 'knngain': supervised evaluation of the quality of a low-dimensional embedding, as introduced in [6]. This function enables computing criteria related to the accuracy of a KNN classifier in the low-dimensional space. The documentation of the function explains the meaning of the criteria and how to interpret them.
- 'viz_2d_emb' and 'viz_qa': visualization of a 2-D embedding and of the quality criteria. These functions respectively enable to: 
---> 'viz_2d_emb': plot a 2-D embedding. 
---> 'viz_qa': depict the quality criteria computed by 'eval_dr_quality' and 'knngain'.
The documentations of the functions describe their parameters. The demo shows how they can be used. 

Notations:
- DR: dimensionality reduction.
- HD: high-dimensional.
- LD: low-dimensional.
- HDS: HD space.
- LDS: LD space.
- NA: Not Available, synonym of missing data, missing values and missing entry.
- NAN: Not A Number, synonym of missing data, missing values and missing entry.
- Ms SNE: multi-scale stochastic neighbor embedding.

References:
[1] de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2019). Nonlinear Dimensionality Reduction With Missing Data Using Parametric Multiple Imputations. IEEE transactions on neural networks and learning systems, 30(4), 1166-1179.
[2] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
[3] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
[4] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.
[5] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.
[6] de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).
[7] Eirola, E., Doquire, G., Verleysen, M., & Lendasse, A. (2013). Distance estimation in numerical data sets with missing values. Information Sciences, 240, 115-128.
[8] Eirola, E., Lendasse, A., Vandewalle, V., & Biernacki, C. (2014). Mixture of gaussians for distance estimation with missing data. Neurocomputing, 131, 32-42.
[9] Sovilj, D., Eirola, E., Miche, Y., Björk, K. M., Nian, R., Akusok, A., & Lendasse, A. (2016). Extreme learning machine for missing data using multiple imputations. Neurocomputing, 174, 220-231.
[10] Bouveyron, C., Girard, S., & Schmid, C. (2007). High-dimensional data clustering. Computational Statistics & Data Analysis, 52(1), 502-519.
[11] Ghahramani, Z., & Jordan, M. I. (1995). Learning from incomplete data.
[12] Rubin, D. B. (2004). Multiple imputation for nonresponse in surveys (Vol. 81). John Wiley & Sons.
[13] Little, R. J., & Rubin, D. B. (2014). Statistical analysis with missing data. John Wiley & Sons.
[14] Cattell, R. B. (1966). The scree test for the number of factors. Multivariate behavioral research, 1(2), 245-276.
[15] Van Hulse, J., & Khoshgoftaar, T. M. (2014). Incomplete-case nearest neighbor imputation in software measurement data. Information Sciences, 259, 596-610.

author: Cyril de Bodt (ICTEAM - UCLouvain)
@email: cyril __dot__ debodt __at__ uclouvain.be
Last modification date: January 7th, 2020
Copyright (c) 2020 Université catholique de Louvain (UCLouvain), ICTEAM. All rights reserved.

The dr_missing_data.py code was tested with Python 3.7.5 (Anaconda distribution, Continuum Analytics, Inc.). It uses the following modules:
- numpy: version 1.17.4 tested
- numba: version 0.46.0 tested
- scipy: version 1.3.2 tested
- matplotlib: version 3.1.1 tested
- scikit-learn: version 0.22 tested
- pandas: version 0.25.3 tested

You can use, modify and redistribute this software freely, but not for commercial purposes. 
The use of this software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.

