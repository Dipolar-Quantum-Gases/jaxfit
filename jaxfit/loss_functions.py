from jax.config import config
print('bs1')
config.update("jax_enable_x64", True)
print('bs2')
import jax.numpy as jnp
print('bs3')
from jax import jit
print('bs4')



class LossFunctionsJIT():

    def __init__(self):
        self.stack_rhos = self.create_stack_rhos()

        self.create_huber_funcs()
        self.create_soft_l1_funcs()
        self.create_cauchy_funcs()
        self.create_arctan_funcs()

        self.IMPLEMENTED_LOSSES = dict(linear=None, huber=self.huber,
                                       soft_l1=self.soft_l1,
                                       cauchy=self.cauchy,
                                       arctan=self.arctan)
        self.loss_funcs = self.construct_all_loss_functions()

        self.create_zscale()
        self.create_calculate_cost()
        self.create_scale_rhos()

    def create_stack_rhos(self):
        @jit
        def stack_rhos(rho0, rho1, rho2):
            return jnp.stack([rho0, rho1, rho2])

        return stack_rhos

    def get_empty_rhos(self, z):
        dlength = len(z)
        rho1 = jnp.zeros([dlength])
        rho2 = jnp.zeros([dlength])
        return rho1, rho2

    def create_huber_funcs(self):
        @jit
        def huber1(z):
            mask = z <= 1

            return jnp.where(mask, z, 2 * z ** 0.5 - 1), mask

        @jit
        def huber2(z, mask):
            rho1 = jnp.where(mask, 1, z ** -0.5)
            rho2 = jnp.where(mask, 0, -0.5 * z ** -1.5)
            return rho1, rho2

        self.huber1 = huber1
        self.huber2 = huber2

    def huber(self, z, cost_only):
        rho0, mask = self.huber1(z)
        if cost_only:
            rho1, rho2 = self.get_empty_rhos(z)
        else:
            rho1, rho2 = self.huber2(z, mask)
        return self.stack_rhos(rho0, rho1, rho2)

    def create_soft_l1_funcs(self):
        @jit
        def soft_l1_1(z):
            t = 1 + z
            return 2 * (t ** 0.5 - 1), t

        @jit
        def soft_l1_2(t):
            rho1 = t ** -0.5
            rho2 = -0.5 * t ** -1.5
            return rho1, rho2

        self.soft_l1_1 = soft_l1_1
        self.soft_l1_2 = soft_l1_2

    def soft_l1(self, z, cost_only):
        rho0, t = self.soft_l1_1(z)
        if cost_only:
            rho1, rho2 = self.get_empty_rhos(z)
        else:
            rho1, rho2 = self.soft_l1_2(t)
        return self.stack_rhos(rho0, rho1, rho2)

    def create_cauchy_funcs(self):
        @jit
        def cauchy1(z):
            return jnp.log1p(z)

        @jit
        def cauchy2(z):
            t = 1 + z
            rho1 = 1 / t
            rho2 = -1 / t ** 2
            return rho1, rho2

        self.cauchy1 = cauchy1
        self.cauchy2 = cauchy2

    def cauchy(self, z, cost_only):
        rho0 = self.cauchy1(z)
        if cost_only:
            rho1, rho2 = self.get_empty_rhos(z)
        else:
            rho1, rho2 = self.cauchy2(z)
        return self.stack_rhos(rho0, rho1, rho2)

    def create_arctan_funcs(self):
        @jit
        def arctan1(z):
            return jnp.arctan(z)

        @jit
        def arctan2(z):
            t = 1 + z ** 2
            return 1 / t, -2 * z / t ** 2

        self.arctan1 = arctan1
        self.arctan2 = arctan2

    def arctan(self, z, cost_only):
        rho0 = self.arctan1(z)
        if cost_only:
            rho1, rho2 = self.get_empty_rhos(z)
        else:
            rho1, rho2 = self.arctan2(z)
        return self.stack_rhos(rho0, rho1, rho2)

    def create_zscale(self):
        @jit
        def zscale(f, f_scale):
            return (f / f_scale) ** 2

        self.zscale = zscale

    def create_calculate_cost(self):
        @jit
        def calculate_cost(f_scale, rho, data_mask):
            cost_array = jnp.where(data_mask, rho[0], 0)
            return 0.5 * f_scale ** 2 * jnp.sum(cost_array)

        self.calculate_cost = calculate_cost

    def create_scale_rhos(self):
        @jit
        def scale_rhos(rho, f_scale):
            rho0 = rho[0] * f_scale ** 2
            rho1 = rho[1]
            rho2 = rho[2] / f_scale ** 2
            return self.stack_rhos(rho0, rho1, rho2)

        self.scale_rhos = scale_rhos

    def construct_single_loss_function(self, loss):
        def loss_function(f, f_scale, data_mask=None, cost_only=False):
            z = self.zscale(f, f_scale)
            rho = loss(z, cost_only=cost_only)
            if cost_only:
                return self.calculate_cost(f_scale, rho, data_mask)
            rho = self.scale_rhos(rho, f_scale)
            return rho

        return loss_function

    def construct_all_loss_functions(self):
        loss_funcs = {}
        for key, loss in self.IMPLEMENTED_LOSSES.items():
            loss_funcs[key] = self.construct_single_loss_function(loss)

        return loss_funcs

    def get_loss_function(self, loss):
        if loss == 'linear':
            return None

        if not callable(loss):
            return self.loss_funcs[loss]
        else:
            def loss_function(f, f_scale, data_mask=None, cost_only=False):
                z = self.zscale(f, f_scale)
                rho = loss(z)
                if cost_only:
                    return self.calculate_cost(f_scale, rho, data_mask)
                rho = self.scale_rhos(rho, f_scale)
                return rho

        return loss_function
