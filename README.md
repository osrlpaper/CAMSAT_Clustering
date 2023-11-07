# CAMSAT clustering for speaker recognition/verification

This repository is for our paper entitled "CAMSAT: Augmentation Mix and Self-Augmented Training Clustering for Self-Supervised Speaker Recognition" currently under review.

Code of our experiments as well as all pseudo-labels and trained models will be available upon acceptance of our paper. 

- The general process for training our clustering generated pseudo-label-based self-supervised speaker embedding networks: ![](/process_pseudo_label_based_speaker_embedding_training.png)

- The pipeline of our proposed CAMSAT clustering method depicting the data flow and the different losses employed for clustering: ![](/CAMSAT_diagram.png)
  We constrain the predictions of the MLP-based neural network to remain unchanged under local perturbations and data augmentations while enforcing symmetry w.r.t. augmentations (red arrows). Information maximization refers to maximizing the information-theoretic dependency between data and their assignments (predictions).

- The evolution of clustering metrics over epochs and the number of clusters discovered during training of our CAMSAT versus other clustering systems based on various loss combinations: ![](/metrics_overtime.png)
  Results show clearly that our proposed additional supervisory signal (mix of predictions of augmented samples) through $ L_{symmetry} $ loss term is crucial for better estimation/discovery of the ground-truth number of clusters. Indeed, despite a predefined 10000 number of clusters, all CAMSAT-based systems were better able to converge to a number of clusters close to the 5994 ground-truth number, on the contrary to all other systems that stick to a number close to the initially predefined number. Additionally, we can observe that $ L_{symmetry} $ has a regularization effect which delays the convergence of clustering metrics over time, while ensuring a steady convergence. We believe this phenomenon can be attributed to the good effect of mitigation of noise memorization which helps the model to focus on the most salient features first by providing it with more time to discover the most relevant ones, instead of memorizing spurious features that generalize less. This can be confirmed, throughout our experiments, by the better downstream EER performance of our CAMSAT-based speaker verification systems (also see our table below) compared to all other studied systems.
