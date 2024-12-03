# Optimal 3DFSE

Code for *Deep Learning Accelerated 3D T2-FLAIR Imaging with Learned K-space Sampling*

In this work, we made sampling coordiates trainable parameters, simulated sampling on exisiting pseudo groud truth images, reconstructed the simulated k-space data with unrolled model-based deep learning and learned the optimal sampling pattern. We used a straight line readout.
The sampling coordinates were not confined on grids, hence the need for NUFFT.


## Backpropagation w.r.t. coordinates

We adapted code from [Bindings-NUFFT-pytorch](https://github.com/albangossard/Bindings-NUFFT-pytorch) and [Bjork](https://github.com/guanhuaw/Bjork) and reimplemented with NUFFT operator from [Sigpy](https://github.com/mikgroup/sigpy).
This allows for accurate gradient calculation by Jacobians [1-3] for multi-channel 3D sampling.


## Description

`train.py`: jointly trains the sampling coordinates and unrolled DL reconstruction. It requires pseudo ground truth images acquired with standard T2-prepped 3DFSE and reconstructed with compressed sensing.
`model.py`: contains the simple blockwise denoiser we used in the unrolled DL reconstruction.
`recon.py`: reconstructs prosepctive scans acquired with learned optimal sampling. It needs raw k-space data and the sampling coordinates txt files, which can be provided upon request.


## References
[1] Wang, G., & Fessler, J. A. (2023). Efficient approximation of Jacobian matrices involving a non-uniform fast Fourier transform (NUFFT). IEEE transactions on computational imaging, 9, 43-54.

[2] A. Gossard, F. de Gournay and P. Weiss. Off-the-grid data-driven optimization of sampling schemes in MRI. In international Traveling Work-shop on Interactions between low-complexity data models and Sensing Techniques (iTWIST) 2020.

[3] A. Gossard, F. de Gournay and P. Weiss. Bayesian Optimization of Sampling Densities in MRI. arXiv preprint arXiv:2209.07170, 2022.

