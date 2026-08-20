"""7-DoF Panda inverse-optimal-control experiments.

`problem`  the scene model, the trajectory parameterization, feature whitening
           and the demonstration/screening pipeline shared by E1-E3
`bases`    the cost bases: kinematic (K=3), per-joint variants (K=9, 16) and the
           dynamic basis whose torque feature runs on GRiD CUDA
`e1_identifiability`  recovery vs demonstration count and demonstration noise
`e2_scaling`          cost of a fit vs cost dimension K
`e3_dynamics`         the price of a misspecified (kinematic) cost basis
"""
