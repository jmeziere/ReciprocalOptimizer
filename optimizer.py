"""
Defines a class for solving the BCDI problem in reciprocal space, rather
than real space.
"""

# pylint: disable = no-name-in-module
from torch import (
    exp,
    sum as tsum,
    abs as tabs,
    max as tmax,
    min as tmin,
    bool as tbool,
    imag,
    conj,
    angle,
    arange,
    meshgrid,
    zeros,
    float32,
    float64,
    rand,
    pi,
)

# pylint: enable = no-name-in-module
from torch.optim import LBFGS
from torch.fft import fftn, ifftn
from torch.cuda import is_available

# pylint: disable = not-callable


class ReciprocalOptimizer:
    """
    ReciprocalOptimizer focuses on solving the BCDI problem using optimization
    in reciprocal space rather than real space. This is done through two steps.
    First, a gradient optimization method is used to minimize the error of the
    real space object w.r.t. the phase of the diffaction pattern. The error in
    this case is the L_2 norm with the "correct" values being either zero or
    one (i.e. inside or outside of the object). Computationally, this is done
    with a mask. The second phase of the optimization problem is shrinkwrap,
    which updates the mask after the gradient optimization.
    """

    def __init__(self, mod, shrink_thresh, shrink_sigma, use_gpu=True):
        if is_available() and use_gpu:
            self.torch_type = float32
            device = "cuda"
        else:
            self.torch_type = float32
            device = "cpu"
        self.mod = mod.to(device)
        self.theta = 2 * pi * rand(mod.shape)

        self.mask = zeros(mod.shape, dtype=tbool).to(device)

        kernel_dims = []
        for s in self.mod.shape:
            kernel_dims.append(tmin(arange(s), arange(s, 0, -1)))

        kernel_mesh = meshgrid(*kernel_dims, indexing="ij")
        real_kernel = (
            exp(-sum(mesh**2 for mesh in kernel_mesh) / (2 * shrink_sigma**2))
        ).to(device)
        self.shrink_kernel = fftn(real_kernel)
        self.shrink_thresh = shrink_thresh

        self.shrinkwrap(use_theta=False)

        self.optimizer = LBFGS(
            [self.theta],
            history_size=10,
            max_iter=100,
            line_search_fn="strong_wolfe",
        )

    ################################################################
    # Begin Optimization Procedures
    ################################################################

    def objective(self):
        """
        The objective function for the optimization problem in reciprocal space.
        Uses the L_2 norm with the "correct" values being either zero or one.
        """
        mask = self.mask.to(self.torch_type)
        recip_space = self.mod * exp(1j * self.theta)
        real_space = ifftn(recip_space, norm="ortho")

        c = tsum(tabs(real_space) * mask) / tsum(tabs(real_space) ** 2)

        error = tsum((c * tabs(real_space) - mask) ** 2)
        derror_dreal_space = (
            2 * c * (c * real_space - mask * exp(1j * angle(real_space)))
        )
        self.theta.grad = imag(
            fftn(derror_dreal_space, norm="ortho") * conj(recip_space)
        )
        return error

    def closure(self):
        """
        The closure for the optimization problem, pytorch style
        """
        self.optimizer.zero_grad()
        objective_eval = self.objective()
        return objective_eval

    def optimize(self):
        """
        Run the optimization problem
        """
        self.optimizer = LBFGS(
            [self.theta],
            history_size=10,
            max_iter=100,
            line_search_fn="strong_wolfe",
        )

        for _ in range(100):
            self.optimizer.step(self.closure)

    ################################################################
    # End Optimization Procedures
    ################################################################

    ################################################################
    # Begin Shrinkwrap
    ################################################################

    def shrinkwrap(self, use_theta=True):
        """
        Update the mask with shrinkwrap
        """
        if use_theta:
            real_space = tabs(ifftn(self.mod * exp(1j * self.theta)))
        else:
            real_space = tabs(ifftn(self.mod))
        real_space_shrink = ifftn(fftn(real_space) * self.shrink_kernel)
        self.mask = tabs(real_space_shrink) > self.shrink_thresh * tmax(
            tabs(real_space_shrink)
        )

    ################################################################
    # End Shrinkwrap
    ################################################################
