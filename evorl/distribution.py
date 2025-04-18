from typing import Any

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions  # note: tfp use lazy init.


def get_categorical_dist(logits: jax.Array):
    """Get a categorical distribution."""
    return tfd.Categorical(logits=logits)


def get_tanh_norm_dist(loc: jax.Array, scale: jax.Array, min_scale: float = 1e-3):
    """Get a tanh transformed normal distribution."""
    scale = jax.nn.softplus(scale) + min_scale
    distribution = tfd.Normal(loc=loc, scale=scale)
    return tfd.Independent(
        TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1
    )


# class TanhNormal(distrax.Transformed):
#     def __init__(self, loc, scale):
#         super().__init__(
#             distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale),
#             distrax.Block(distrax.Tanh(), ndims=1)
#         )

#     def mode(self):
#         loc = self.distribution.mode()
#         return self.bijector.forward(loc)

#     def entropy(self, input_hint=None):
#         """
#             No analytical form. use sample to estimate.
#             input_hint: an example sample from the base distribution
#                 eg: self.distribution.sample(seed=jax.random.PRNGKey(42))
#         """

#         return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
#             input_hint)


def get_trancated_norm_dist(loc, scale, low, high):
    """Get a truncated normal distribution."""
    return tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """Distribution followed by tanh. from acme."""

    def __init__(self, distribution, threshold=0.999, validate_args=False):
        """Initialize the distribution.

        Args:
          distribution: The distribution to transform.
          threshold: Clipping value of the action when computing the logprob.
          validate_args: Passed to super class.
        """
        super().__init__(
            distribution=distribution,
            bijector=tfp.bijectors.Tanh(),
            validate_args=validate_args,
        )
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(1.0 - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = (
            self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        )
        self._log_prob_right = (
            self.distribution.log_survival_function(inverse_threshold) - log_epsilon
        )

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold,
            self._log_prob_left,
            jnp.where(
                event >= self._threshold, self._log_prob_right, super().log_prob(event)
            ),
        )

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed=None):
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0
        )

    @classmethod
    def _parameter_properties(cls, dtype: Any | None, num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
