# Project Title
RMSD (root mean squared deviation) loss computation using pytorch.

## What it does?
It computes batch-wise RMSD loss using pytorch and quaternion math. It also translate the 3d coordinates before computing squared deviation. The code is more or less commented to understand what is going on. Moreover, special thanks to "Pande Lab at Stanford University", who implements RMSD loss using Tensorflow [1][2]. In this implemetation, I just followed their easy steps to leverage pytorch autograd functionality.

## Requirements
Python 3
Pytorch

## How to run?
An example is: 
```
y = torch.randn((30, 256, 3)) # ground truth 3d coordinates
y_prime = torch.randn((30, 256, 3), requires_grad=True) # predicted 3d coordinates
# y_prime, y: (batch_size, n_coords, 3) 
rmsd_loss = RMSD_loss()
loss = rmsd_loss(y_prime, y)
print(loss)
```

## References
1. https://towardsdatascience.com/tensorflow-rmsd-using-tensorflow-for-things-it-was-not-designed-to-do-ada4c9aa0ea2
2. https://github.com/mdtraj/tftraj/blob/master/tftraj/rmsd.py
    
