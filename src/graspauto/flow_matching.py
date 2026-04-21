"""Conditional Flow Matching kernel and ODE samplers.

Implements rectified-flow style conditional flow matching (Lipman et al. 2023,
"Flow Matching for Generative Modeling"; Liu et al. 2023, "Flow Straight and
Fast: Learning to Generate and Transfer Data with Rectified Flow") with a
linear interpolation kernel.

Mathematical setup:

- Two distributions: a base π₀ (here: standard Gaussian) and a target π₁
  (here: empirical samples from training data).
- Linear interpolation: xt = (1−t)·x₀ + t·x₁ for t ∈ [0, 1].
- Target velocity: u(xt, t) = x₁ − x₀  (the constant velocity along the
  straight line from x₀ to x₁).
- Training: a velocity network vθ(xt, t | c) is trained to predict u, given
  a sampled (x₀, x₁, t, condition c) tuple. Loss is L = ‖vθ(xt, t|c) − (x₁−x₀)‖².
- Sampling: integrate dxt/dt = vθ(xt, t | c) from t=0 (x₀ ∼ π₀) to t=1
  (yielding x₁, the generated sample), via Euler or RK4 ODE integration.

This module is **dependency-free** beyond PyTorch — no MANO, no project-
specific data shapes. The velocity field is passed in as a callable so the
same kernel works for any conditional generative task. This design lets us
unit-test the kernel on toy 2D distributions before wiring up the full
graspauto pipeline.

References:
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023).
  Flow Matching for Generative Modeling. ICLR 2023.
- Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Learning to
  Generate and Transfer Data with Rectified Flow. ICLR 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import torch
from torch import Tensor


VelocityFn = Callable[[Tensor, Tensor, Optional[Any]], Tensor]
"""Signature for a velocity field callable.

Args:
    xt:        Tensor of shape (B, D) — current state.
    t:         Tensor of shape (B,)  — flow time in [0, 1] for each sample.
    condition: Anything the network needs (cross-attention inputs, conditioning
               tokens, etc.). May be None for unconditional flows.

Returns:
    Tensor of shape (B, D) — predicted velocity.
"""


@dataclass(frozen=True)
class FlowMatchingTrainingBatch:
    """A single training batch produced by `ConditionalFlowMatching.prepare_batch`.

    Attributes:
        xt:           Interpolated state at time t. Shape (B, D).
        t:            Per-sample flow time in [0, 1]. Shape (B,).
        target_velocity: The "ground truth" velocity the network must predict.
                         For linear interpolation kernel, this equals (x1 − x0).
                         Shape (B, D).
        x0:           The base sample. Shape (B, D).
        x1:           The target sample. Shape (B, D).
    """

    xt: Tensor
    t: Tensor
    target_velocity: Tensor
    x0: Tensor
    x1: Tensor


class ConditionalFlowMatching:
    """Stateless conditional flow matching kernel with linear interpolation.

    The kernel itself holds no learnable parameters. The learnable component
    is the user-provided velocity field, which is passed in at training and
    sampling time.
    """

    def __init__(self, sigma_min: float = 0.0):
        """Initialize the flow matching kernel.

        Args:
            sigma_min: Optional small noise added to xt to prevent the
                interpolation from collapsing to a delta at t=0 or t=1.
                Set to 0.0 for the canonical rectified-flow / linear
                interpolation. Set to a small positive value (e.g. 1e-3)
                for added robustness during training.
        """
        if sigma_min < 0.0:
            raise ValueError(f"sigma_min must be non-negative, got {sigma_min}")
        self.sigma_min = float(sigma_min)

    # ------------------------------------------------------------------
    # Training side: build a (xt, t, target velocity) triple from (x0, x1)
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        x0: Tensor,
        x1: Tensor,
        *,
        t: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> FlowMatchingTrainingBatch:
        """Sample t uniformly in [0, 1] and build the interpolated state + target.

        Args:
            x0: Base distribution sample, shape (B, D).
            x1: Target distribution sample, shape (B, D).
            t:  Optional pre-sampled time tensor of shape (B,). If None,
                sampled uniformly from [0, 1].
            generator: Optional torch.Generator for reproducible t sampling.

        Returns:
            A `FlowMatchingTrainingBatch`.
        """
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 must have matching shapes, got {x0.shape} vs {x1.shape}"
            )
        if x0.dim() != 2:
            raise ValueError(
                f"x0 and x1 must be 2-D (B, D), got {x0.dim()}-D"
            )

        batch_size = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        if t is None:
            t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        else:
            if t.shape != (batch_size,):
                raise ValueError(f"t must have shape ({batch_size},), got {tuple(t.shape)}")
            t = t.to(device=device, dtype=dtype)

        # Linear interpolation: xt = (1 - t) * x0 + t * x1
        # Broadcasting: t is (B,), expand to (B, 1) so it broadcasts over the D axis.
        t_b = t.unsqueeze(-1)
        xt = (1.0 - t_b) * x0 + t_b * x1

        if self.sigma_min > 0.0:
            noise = torch.randn(x0.shape, device=device, dtype=dtype, generator=generator)
            xt = xt + self.sigma_min * noise

        target_velocity = x1 - x0

        return FlowMatchingTrainingBatch(
            xt=xt,
            t=t,
            target_velocity=target_velocity,
            x0=x0,
            x1=x1,
        )

    def loss(
        self,
        predicted_velocity: Tensor,
        target_velocity: Tensor,
        *,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> Tensor:
        """Standard L2 flow-matching loss.

        Args:
            predicted_velocity: Output of the velocity network at (xt, t, c).
                Shape (B, D).
            target_velocity:    Target from `prepare_batch`. Shape (B, D).
            reduction:          'mean' (default), 'sum', or 'none'.

        Returns:
            Scalar tensor (or per-sample tensor if reduction='none').
        """
        if predicted_velocity.shape != target_velocity.shape:
            raise ValueError(
                f"predicted and target velocity shapes must match, "
                f"got {predicted_velocity.shape} vs {target_velocity.shape}"
            )

        per_element = (predicted_velocity - target_velocity) ** 2
        per_sample = per_element.mean(dim=-1)  # mean over D, keeping B

        if reduction == "mean":
            return per_sample.mean()
        if reduction == "sum":
            return per_sample.sum()
        if reduction == "none":
            return per_sample
        raise ValueError(f"unknown reduction: {reduction!r}")

    # ------------------------------------------------------------------
    # Sampling side: integrate the ODE dxt/dt = vθ(xt, t | c) from t=0 to t=1
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        velocity_fn: VelocityFn,
        x0: Tensor,
        *,
        condition: Optional[Any] = None,
        num_steps: int = 10,
        method: Literal["euler", "rk4"] = "rk4",
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Generate a sample by ODE-integrating the learned velocity field.

        Integrates dxt/dt = velocity_fn(xt, t, condition) from t=0 to t=1
        starting from x0. Uses either explicit Euler or 4th-order Runge–Kutta.

        Args:
            velocity_fn: Callable matching the `VelocityFn` signature.
            x0:          Base distribution sample, shape (B, D).
            condition:   Whatever conditioning the velocity_fn expects, or None.
            num_steps:   Number of integration steps. RK4 default 10 ≈ Euler 100.
            method:      'euler' or 'rk4'.
            return_trajectory: If True, also return the full trajectory of
                shape (num_steps + 1, B, D) — useful for visualization and
                debugging. If False (default), return only the final x1.

        Returns:
            Final state x1 of shape (B, D). If return_trajectory=True, also
            returns the trajectory of shape (num_steps + 1, B, D).
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if x0.dim() != 2:
            raise ValueError(f"x0 must be 2-D (B, D), got {x0.dim()}-D")

        batch_size = x0.shape[0]
        device = x0.device
        dtype = x0.dtype
        dt = 1.0 / num_steps

        xt = x0.clone()
        if return_trajectory:
            trajectory = [xt.clone()]

        if method == "euler":
            for step in range(num_steps):
                t_val = step * dt
                t_tensor = torch.full((batch_size,), t_val, device=device, dtype=dtype)
                v = velocity_fn(xt, t_tensor, condition)
                xt = xt + dt * v
                if return_trajectory:
                    trajectory.append(xt.clone())
        elif method == "rk4":
            for step in range(num_steps):
                t_val = step * dt
                t1 = torch.full((batch_size,), t_val, device=device, dtype=dtype)
                k1 = velocity_fn(xt, t1, condition)

                t2 = torch.full((batch_size,), t_val + 0.5 * dt, device=device, dtype=dtype)
                k2 = velocity_fn(xt + 0.5 * dt * k1, t2, condition)

                t3 = t2  # same time as k2
                k3 = velocity_fn(xt + 0.5 * dt * k2, t3, condition)

                t4 = torch.full((batch_size,), t_val + dt, device=device, dtype=dtype)
                k4 = velocity_fn(xt + dt * k3, t4, condition)

                xt = xt + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                if return_trajectory:
                    trajectory.append(xt.clone())
        else:
            raise ValueError(f"unknown integration method: {method!r}")

        if return_trajectory:
            return xt, torch.stack(trajectory, dim=0)
        return xt

    @torch.no_grad()
    def sample_unconditional(
        self,
        velocity_fn: VelocityFn,
        *,
        batch_size: int,
        dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        num_steps: int = 10,
        method: Literal["euler", "rk4"] = "rk4",
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Convenience: sample a fresh x0 from N(0, I) and integrate.

        Equivalent to:
            x0 = torch.randn(batch_size, dim, ...)
            return self.sample(velocity_fn, x0, condition=None, num_steps=num_steps, method=method)
        """
        x0 = torch.randn(
            batch_size, dim,
            device=torch.device(device),
            dtype=dtype,
            generator=generator,
        )
        return self.sample(
            velocity_fn,
            x0,
            condition=None,
            num_steps=num_steps,
            method=method,
        )
