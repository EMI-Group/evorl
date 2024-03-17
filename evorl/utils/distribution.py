import jax
import distrax
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
# import tensorflow_probability.substrates.jax.distributions as tfd

# import types
# def get_tanh_norm_dist(loc: jax.Array, scale: jax.Array):
#     dist = distrax.Transformed(
#         distrax.Normal(loc=loc, scale=scale),
#         distrax.Tanh()
#     )

#     def tanh_norm_mode(dist):
#         loc = dist.distribution.mode()
#         return dist.bijector.forward(loc)
    
#     dist.mode = types.MethodType(tanh_norm_mode, dist)
#     return dist
    

class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        super().__init__(
            distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale),
            distrax.Block(distrax.Tanh(), ndims=1)
        )

    def mode(self):
        loc = self.distribution.mode()
        return self.bijector.forward(loc)
    
    def entropy(self, input_hint = None):
        """
            No analytical form. use sample to estimate.
        """
        # entropy = self.distribution.entropy()
        # # if input_hint is None:
        # #     input_hint = self.distribution.sample()
        # entropy += self.bijector.forward_log_det_jacobian(
        #     input_hint
        # )

        # return entropy.sum(axis=-1)
        raise NotImplementedError()



def get_trancated_norm_dist(loc, scale, low, high):
    return tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)

