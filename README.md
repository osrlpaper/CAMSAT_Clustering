# CAMSAT clustering for speaker recognition/verification

This repository is for our paper entitled "CAMSAT: Augmentation Mix and Self-Augmented Training Clustering for Self-Supervised Speaker Recognition" currently under review.

Code of our experiments as well as all pseudo-labels and trained models will be available upon acceptance of our paper. 

## Our general framework and CAMSAT

- The general process for training our clustering generated pseudo-label-based self-supervised speaker embedding networks: ![](/process_pseudo_label_based_speaker_embedding_training.png)

- The pipeline of our proposed CAMSAT clustering method depicting the data flow and the different losses employed for clustering: ![](/CAMSAT_diagram.png)
  We constrain the predictions of the MLP-based neural network to remain unchanged under local perturbations and data augmentations while enforcing symmetry w.r.t. augmentations (red arrows). Information maximization refers to maximizing the information-theoretic dependency between data and their assignments (predictions).

## Clustering robustness and sensitivity to the predefined number of clusters
- The evolution of clustering metrics over epochs and the number of clusters discovered during training of our CAMSAT versus other clustering systems based on various loss combinations: ![](/metrics_overtime.png)
  Results show clearly that our proposed additional supervisory signal (mix of predictions of augmented samples) through $L_{symmetry}$ loss term is crucial for better estimation/discovery of the ground-truth number of clusters. Indeed, despite a predefined 10000 number of clusters, all CAMSAT-based systems were better able to converge to a number of clusters close to the 5994 ground-truth number, on the contrary to all other systems that stick to a number close to the initially predefined number. Additionally, we can observe that $L_{symmetry}$ has a regularization effect which delays the convergence of clustering metrics over time, while ensuring a steady convergence. We believe this phenomenon can be attributed to the good effect of mitigation of noise memorization which helps the model to focus on the most salient features first by providing it with more time to discover the most relevant ones, instead of memorizing spurious features that generalize less. This can be confirmed, throughout our experiments, by the better downstream EER performance of our CAMSAT-based speaker verification (SV) systems (also see our table below) compared to all other studied systems.

- Comparison of the evolution of clustering metrics over epochs and the number of clusters discovered during training of our CAMSAT trained with different initial numbers of clusters: ![](/metrics_overtime_camsat.png)
We can observe that our CAMSAT method is not sensitive to the initial number of clusters and is able to estimate the ground truth number almost perfectly.

## Correlation between clustering performance (clustering metrics) and downstream performance
In the following experiments, and inspired by the work of [1] to analyze the generalizability of pseudo-labels (PLs), we follow the same setup to analyze the correlation of our various generated clustering metrics and the downstream speaker verification EER validation performance. Our experiments encompass 122 different clustering-based pseudo-labels in total. In this section, we also extend our experiments to additionally incorporate two variants of mixup at both the instance input-level (i-mix) [3] and the latent space (l-mix) [2] as regularizations techniques in order to improve the generalization of our SV systems. This also allows us to extend their previous small-scale experiment to include our large number of 122 different clustering systems and explore in a more comprehensive way the effectiveness of mixup to reduce
the memorization effects of noisy labels.

- Heatmap plot of the Pearson product-moment correlation coefficients between the clustering metrics and the speaker verification EER validation performance for all pseudo-labels combined (122 different pseudo-labels in total): ![](/eer_heatmap_all_systems.png)

We can observe a very high correlation between the various supervised clustering metrics and the downstream task. This also demonstrates that clustering metrics are highly predictive of the final downstream speaker-verification performance. Additionally, we find that notions such as completeness, homogeneity and normalized mututal information (NMI) of generated clustering assignments are very important.
- Heatmap plot of the Pearson product-moment correlation coefficients between the clustering metrics and the speaker verification EER validation performance for classical (non-deep) models versus deep-learning-based clustering models: ![](/eer_heatmap_classical_vs_deep.png)
We observe that correlation with mutual information (through NMI) is more significant for deep clustering models whereas performance of classical models-based speaker verification systems exhibit a higher correlation with the clustering accuracy of pseudo-labels (also purity).

Moreover, results also indicate that unsupervised clustering metrics (Silhouette score, Calinski-Harabasz score (CHS), and Davies-Bouldin score (DBS)) are more predictive and useful to infer the generalizability of pseudo-labels in the case of non-deep clustering models, contrary to deep clustering models. This can probably be explained by the reliance of these models on classical distances such as the euclidean distance where they directly employ the input vectors in computing distances (instead of further processing these inputs to generate higher abstract features as is the case with deep learning models). We believe this makes those  unsupervised metrics highly correlated with the final dosnstream performance. As for deep neural networks-based clustering models, these metrics seem to be less useful to predict the generalizability of the generated pseudo-labels. In the case of deep clustering models, we believe concepts used by these metrics such as between-cluster dispersion or within-cluster dispersion using euclidean distance, should be applied on the generated representations instead of applying them directly on the input features which showed considerable limitations.
As a consequence, and contrary to deep clustering models, our analysis of the correlations corresponding to the 3 unsupervised clustering metrics demonstrates that classical models seem to suffer from a serious shortcoming: they are less capable to analyze and extract abstract features from the data inputs, and seem to rely more on simple distances computed directly on the input features instead of higher abstract features/concepts and structure. 

In the following 2 experiments, we try to analyze the effect of noise, both at the input-level (noisy features) and at the level of pseudo-labels on the generalization of clustering-based self-supervised-based speaker verification approaches.

- Correlation between clustering metrics and downstream EER validation performance for clustering systems using noisy input samples (employing mean and standard deviation scaling of input vectors along the features axis) versus systems using clean input samples (employing l2-norm independent normalization along the samples axis): ![](/eer_heatmap_stdScaler_vs_l2Norm.png)

  We can observe that when using clean input samples during clustering and without using any other type of regularization (in our case we study i-mix and l-mix), the performance of our speaker verification systems (No Reg. column) shows a higher correlation with the clustering metrics than when training these clustering models with noisy samples. Additionally, the higher correlation of performance of the noisy samples-based systems with the Silhouette score shows that clustering-based self-supervised-based speaker verification approaches can suffer a considerable problem of generalization when data is noisy.
  
- Correlation between clustering metrics and downstream EER validation performance for highly accurate pseudo-labels-based clustering systems (above 50% unsupervised clustering accuracy) versus noisy pseudo-labels-based clustering systems (below 50% accuracy): ![](/eer_heatmap_50percentACC.png)

Similar to our experiments above dealing with input noise, here we study the effect of training clustering-based self-supervised-based speaker verification systems with very noisy pseudo-labels (below 50% ACC) compared to systems trained with more accurate pseudo-labels. Our experiments show that label noise leads to lower correlation of the clustering performance with the downstream SV performance (lower coefficients when pseudo-labels are less accurate) which highlights the importance of highly accurate clustering models and the importance of mitigating label noise. From the heatmap, we can also notice that in the case of highly noisy pseudo-labels, i-mix and l-mix performing mixup regularization exhibit higher correlation with the downstream performance, which indicates the effectiveness of these 2 regularization techniques to mitigate the problem of noise memorization and to induce better generalization of our SV systems. We find these results to be compatible with [1] where authors demonstrated a high effectiveness of mixup, at both input (i-mix) and latent space levels (l-mix), to mitigate the memorization effects of noisy pseudo-labels and prevent overfitting inaccurate pseudo-labels.

## Study of various maximum margin-based softmax loss objectives
In order to improve performance on previously unseen data and to generalize to out-of-domain speech samples, in this section we study various maximum margin-based softmax variants based on different
objectives. Indeed, softmax suffers from several drawbacks such as that (1) its computation of inter-class margin is intractable [18] and (2) the learned projections are not guaranteed equi-spaced. Indeed,
the projection vectors for majority classes occupy more angular space compared to minority classes [39]. To solve these problems, several alternatives to softmax have been proposed [14 , 57 , 64 , 35, 58].
For instance, AM Softmax loss applies an additive margin constraint in the angular space to the softmax loss for maximizing inter-class variance and minimizing intra-class variance. To provide a
clear geometric interpretation of data samples and enhance the discriminative power of deep models, AAMSoftmax (angular additive margin softmax) objective (aka ArcFace) introduces an additive
angular margin to the target angle (between the given features and the target center). Due to the exact correspondence between the angle and arc in the normalized hypersphere, AAMSoftmax can directly
optimize the geodesic distance margin, thus its other name ArcFace. Additionally, CosFace (large margin cosine loss) reformulates the softmax loss as a cosine loss by L2 normalizing both features
and weight vectors to remove radial variations, based on which a cosine margin term is introduced to further maximize the decision margin in the angular space. On the other hand, OCSoftmax uses
one-class learning instead of multi-class classification and does not assume the same distribution for all classes/speakers. More recently, AdaFace loss has been proposed which emphasizes misclassified
samples according to the quality of speaker embeddings (via feature norms).

Table below summarizes our results using different predefined numbers of clusters and different clustering-based pseudo-labels.

![](/maximum_margin_softmax_experiments.png)

Our experimental results show clearly that our adopted softmax variants are very effective in improving the generalization of our speaker verification systems. In particular, unlike the widely used
AAMSoftmax loss in speaker verification, to our knowledge, our results indicate for the first time that variants such as OCSoftmax (does not assume the same distribution for all speakers which is more
realistic in our case) or the recent AdaFace loss, perform consistently better across all pseudo-labels and the ground truth labels. Indeed, AAMSoftmax is susceptible to massive label noise [14]. This is
because if a training sample is a noisy sample, it does not belong to the corresponding positive class.
In AAMSoftmax, this noisy sample generates a large wrong loss value, which impairs the model training. This partially explains the underperformance of AAMSoftmax compared to other variants
when using pseudo-labels for training.

[1] Fathan, A.; Alam, J.; and Kang, W. 2022. On the impact of the quality of pseudo-labels on the self-supervised speaker verification task. NeurIPS ENLSP Workshop 2022.

[2] W. H. Kang, J. Alam, and A. Fathan. l-mix: a latent-level instance mixup regularization for robust self-supervised speaker representation learning. IEEE Journal of Selected Topics in Signal Processing, 2022.

[3] K. Lee, Y. Zhu, K. Sohn, C.-L. Li, J. Shin, and H. Lee. i-mix: A domain-agnostic strategy for contrastive representation learning. In ICLR, 2021.

[14] Deng et al. Arcface: Additive angular margin loss for deep face recognition. IEEE Transactions on PAMI, 2021. doi: 10.1109/TPAMI.2021.3087709.

[18] G. F. Elsayed et al. Large margin deep networks for classification, 2018.

[35] M. Kim, A. K. Jain, and X. Liu. Adaface: Quality adaptive margin for face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18750–18759, 2022.

[39] W. Liu, Y. Wen, et al. Large-margin softmax loss for convolutional neural networks. In ICML, volume 2, 2016.

[57] F. Wang et al. Additive margin softmax for face verification. IEEE Signal Processing Letters, 25(7):926–930, 2018.

[58] H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, and W. Liu. Cosface: Large margin cosine loss for deep face recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5265–5274, 2018.

[64] Y. Zhang et al. One-class learning towards synthetic voice spoofing detection. IEEE Signal Processing Letters, 2021.
