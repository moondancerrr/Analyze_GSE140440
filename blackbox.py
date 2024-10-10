import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn.functional as F

from local import (
      ZeroInflatedNegativeBinomial,
      warmup_scheduler,
      new_sc_data,
)

# Modules are the different kinds of response to the
# treatments. Every treatment is assumed to have the
# same number of different responses, typically two.

global K # Number of modules / set by user.
global B # Number of batches / from data.
global N # Number of types / from data.
global D # Number of drugs / from data.
global G # Number of genes / from data.

DEBUG = False # Flip flag before inserting breakpoint.
NUM_SAMPLES = 100

# Use only for debugging.
pyro.enable_validation(not DEBUG)


def model(data, generate=False):
   batch, ctype, drugs, X, mask = data
   device = generate.device if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate.shape[0]

   # dim(OH_B): ncells x B
   OH_B = F.one_hot(batch.to(torch.int64)).to(device, dtype)
   # dim(OH_N): ncells x N
   OH_N = F.one_hot(ctype.to(torch.int64)).to(device, dtype)
   # dim(OH_D): ncells x D
   OH_D = F.one_hot(drugs.to(torch.int64)).to(device, dtype)

   # Variance-to-mean ratio. Variance is modelled as
   # 's * u', where 'u' is mean gene expression and
   # 's' is a positive number with 90% chance of being
   # in the interval 1 + (0.3, 200).
   s = 1. + pyro.sample(
         name = "s",
         # dim: 1 x 1 | .
         fn = dist.LogNormal(
            2. * torch.ones(1,1).to(device, dtype),
            2. * torch.ones(1,1).to(device, dtype)
         )
   )

   # Zero-inflation factor 'pi'. The median is set at 
   # 0.15, with 5% chance that 'pi' is less than 0.01
   # and 5% chance that 'pi' is more than 50%.
   pi = pyro.sample(
         # dim: 1 x 1 | .
         name = "pi",
         fn = dist.Beta(
            1. * torch.ones(1,1).to(device, dtype),
            4. * torch.ones(1,1).to(device, dtype)
         )
   )

   with pyro.plate("K", K):

      # Weight of the transcriptional modules, showing
      # the representation of each module in the
      # transcriptomes of the cells. This is the same
      # prior as in the standard latent Dirichlet
      # allocation.
      alpha = pyro.sample(
            name = "alpha",
            # dim(alpha): 1 x K
            fn = dist.Gamma(
               torch.ones(1,1).to(device, dtype) / K,
               torch.ones(1,1).to(device, dtype)
            )
      )

      # dim(alpha): 1 x 1 x K
      alpha = alpha.unsqueeze(-2)


   with pyro.plate("ncells", ncells):

      # Proportion of the modules in the transcriptomes.
      # This is the same hierarchic model as the standard
      # latent Dirichlet allocation.
      theta = pyro.sample(
            name = "theta",
            # dim(theta): 1 x ncells | K
            fn = dist.Dirichlet(
               alpha # dim: 1 x 1 x K
            )
      )

      # Correction for the total number of reads in the
      # transcriptome. The shift in log space corresponds
      # to a cell-specific scaling of all the genes in
      # the transcriptome. In linear space, the median
      # is 1 by design (average 0 in log space). In
      # linear space, the scaling factor has a 90% chance
      # of being in the window (0.2, 5).
      shift = pyro.sample(
            name = "shift",
            # dim(shift): 1 x ncells | .
            fn = dist.Normal(
               0. * torch.zeros(1,1).to(device, dtype),
               1. * torch.ones(1,1).to(device, dtype)
            )
      )

      # dim(shift): ncells x 1
      shift = shift.squeeze().unsqueeze(-1)


   # Per-gene sampling.
   with pyro.plate("G", G):

      # Dummy plate to fix rightmost shapes in different calls
      # (e.g., depending on whether 'vectorize_particles' is
      # True or False).
      with pyro.plate("1xG", 1):

         # The transcriptional modules are strongly correlated
         # because they correspond to different responses to
         # the same treatment (drug). Only a handful of genes
         # are expected to have a distinct response. Between
         # treatments (drugs), there is no correlation. The
         # average expression in log-space has a standard
         # Gaussian distribution, i.e. there is a 90% chance
         # that the gene has a 5-fold increase or decrease
         # over baseline. Within a treatment, there is a 90%
         # chance that the responses for the same gene are
         # within 25% of each other.

         # 'mu_KD':
         #  | 1.00 0.99 0.99 | 0.00 0.00 ...
         #  | 0.99 1.00 0.99 | 0.00 0.00 ...
         #  | 0.99 0.99 1.00 | 0.00 0.00 ...
         #    ----- K ------
         #    -------  D blocks  --------

         # Intermediate to construct the covariance matrix.
         K_block = (.99 * torch.ones(K,K)).fill_diagonal_(1.)

         cor_KD = torch.block_diag(*([K_block]*D)).to(device, dtype)
         mu_KD = torch.zeros(1,1,K*D).to(device, dtype)

         mod = pyro.sample(
               name = "mod",
               # dim(modules): 1 x G | D*K
               fn = dist.MultivariateNormal(
                  0.00 * mu_KD,
                  0.25 * cor_KD
               )
         )

         # dim(mod): 1 x G x D x K
         mod = mod.view(mod.shape[:-1] + (D,K))

         # The baseline varies over batches and cell types.
         # the correlation is stronger between batches than
         # between cell types. As above, the average
         # expression in log-space has a Gaussian
         # distribution, but it has mean 1 and standard
         # deviation 3, so there is a 90% chance that a gene
         # has between 0 and 400 reads in standard form (i.e,
         # without considering sequencing depth).
         # Between batches, there is a 90% chance that the
         # responses for the same gene are within 25% of each
         # other. Between cells, there is a 90% chance that
         # the responses are within a factor 2 of each other.

         # 'cor_NB':
         #  | 1.00  0.99 0.99 |  0.97  0.97 ...
         #  | 0.99  1.00 0.99 |  0.97  0.97 ...
         #  ----------------------------
         #  | 0.97  0.97 0.99 |  1.00  0.99 ...
         #  | 0.97  0.97 0.99 |  0.99  1.00 ...
         #   ------  B  -----
         #   -----------  N blocks  -----------

         # Intermediate to construct the covariance matrix.
         B_block = (.02 * torch.ones(B,B)).fill_diagonal_(.03)

         cor_NB = .97 + torch.block_diag(*([B_block]*N)).to(device, dtype)
         mu_NB = torch.ones(1,1,N*B).to(device, dtype)

         base = pyro.sample(
               name = "base",
               # dim(base): 1 x G | N*B
               fn = dist.MultivariateNormal(
                  1 * mu_NB, # dim: N*B 
                  3 * cor_NB # dim: N*B x N*B
               )
         )

         # dim(base): 1 x G x N x B
         base = base.view(base.shape[:-1] + (N,B))

      # Attribute modules and baseline to each cell. A cell has
      # exactly one treatment (drug), one batch and one cell type
      # so the 'einsum' functions below just select the term that
      # was sampled above. All the transcriptional modules for a
      # given treatment are present in a cell, with breakdown
      # 'theta', so in this case 'einsum' computes the weighted
      # average of the terms sampled above.

      # dim(mod_n): ncells x G
      nprll = len(mod.shape) == 4 # (not run in parallel)
      mod_n = torch.einsum("ni,ygik,xnk->ng", OH_D, mod, theta) if nprll else \
              torch.einsum("ni,...ygik,...xnk->...ng", OH_D, mod, theta)

      # dim(base_n): ncells x G
      nprll = len(base.shape) == 4 # (not run in parallel)
      base_n = torch.einsum("ni,xgij,nj->ng", OH_N, base, OH_B) if nprll else \
               torch.einsum("ni,...xgij,nj->...ng", OH_N, base, OH_B)

      # dim(u): ncells x G
      u = torch.exp(base_n + mod_n + shift)

      # Parameter 'u' is the average number of reads in the cell
      # and the variance is 's x u'. Parametrize 'r' and 'p' (the
      # values required for the negative binomial) as a function
      # of 'u' and 's'.
      p_ = 1. - 1. / s # dim(p_): 1
      r_ = u / (s - 1) # dim(r_): 1 x G x ncells

   # ---------------------------------------------------------------
   # NOTE: if the variance is assumed to vary with a power 'a', i.e.
   # the variance is 's x u^a', then the equations above become:
   # p_ = 1. - 1. / (s*u^(a-1))
   # r_ = u / (s*u^(a-1) - 1)
   # ---------------------------------------------------------------

      # Make sure that parameters of the ZINB are valid. 'p'
      # must be in (0,1) and 'r' must be positive. Choose a
      # small number 'eps' to keep the values away from the
      # boundaries.

      eps = 1e-6
      p = torch.clamp(p_, min=0.+eps, max=1.-eps)
      r = torch.clamp(r_, min=0.+eps)

      with pyro.plate("ncellsxG", ncells):

         # Observations are sampled from a ZINB distribution.
         Y = pyro.sample(
               name = "Y",
               # dim(X): ncells x G
               fn = ZeroInflatedNegativeBinomial(
                  total_count = r, # dim: ncells x G
                  probs = p,       # dim:          1
                  gate = pi        # dim:          1
               ),
               obs = X,
               obs_mask = mask
         )

   # Return sampled transcriptome and "smoothed" log-estimate.
   return torch.stack((Y, mod_n))


def guide(data=None, generate=False):

   batch, ctype, drugs, X, mask = data
   device = generate.device if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate.shape[0]

   # Posterior distribution of 's'.
   post_s_loc = pyro.param(
         "post_s_loc", # dim: 1 x 1
         lambda: 2 * torch.ones(1,1).to(device, dtype)
   )
   post_s_scale = pyro.param(
         "post_s_scale", # dim: 1 x 1
         lambda: 2 * torch.ones(1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   post_s = pyro.sample(
         name = "s",
         # dim: 1 x 1
         fn = dist.LogNormal(
            post_s_loc,  # dim: 1 x 1
            post_s_scale # dim: 1 x 1
         )
   )

   # Posterior distribution of 'pi'.
   post_pi_0 = pyro.param(
         "post_pi_0", # dim: 1 x 1
         lambda: 1. * torch.ones(1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   post_pi_1 = pyro.param(
         "post_pi_1", # dim: 1 x 1
         lambda: 4. * torch.ones(1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   post_pi = pyro.sample(
         name = "pi",
         # dim: 1 x 1
         fn = dist.Beta(
            post_pi_0, # dim: 1 x 1
            post_pi_1  # dim: 1 x 1
         )
   )

   with pyro.plate("K", K):

      post_alpha = pyro.param(
            "post_alpha", # dim: 1 x K
            lambda: torch.ones(1,K).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )

      alpha = pyro.sample(
            name = "alpha",
            # dim(alpha): 1 x K
            fn = dist.Gamma(
               post_alpha, # dim: 1 x K
               torch.ones(1,1).to(device, dtype)
            )
      )

   with pyro.plate("ncells", ncells):

      # Posterior distribution of 'theta'.
      post_theta_param = pyro.param(
            "post_theta_param",
            lambda: torch.ones(1,ncells,K).to(device, dtype),
            constraint = torch.distributions.constraints.greater_than(0.5)
      )
      post_theta = pyro.sample(
            name = "theta",
            # dim(theta): 1 x ncells | K
            fn = dist.Dirichlet(
               post_theta_param # dim: 1 x ncells x K
            )
      )

      # Posterior distribution of 'shift'.
      post_shift_loc = pyro.param(
            "post_shift_loc",
            lambda: 0 * torch.zeros(1,ncells).to(device, dtype),
      )
      post_shift_scale = pyro.param(
            "post_shift_scale",
            lambda: 1 * torch.ones(1,ncells).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      post_shift = pyro.sample(
            name = "shift",
            # dim: 1 x ncells
            fn = dist.Normal(
               post_shift_loc,  # dim: 1 x ncells
               post_shift_scale # dim: 1 x ncells
            )
      )

   with pyro.plate("G", G):

      with pyro.plate("1xG", 1):

         # Posterior distribution of 'mod'.
         post_mod_loc = pyro.param(
               "post_mod_loc", # dim: 1 x G x D*K
               lambda: 0 * torch.zeros(1,G,K*D).to(device, dtype)
         )
         post_mod_scale = pyro.param(
               "post_mod_scale", # dim: 1 x G x 1
               lambda: .25 * torch.ones(1,G,1).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )

         post_mod = pyro.sample(
               name = "mod",
               # dim: 1 x G | D*K
               fn = dist.Normal(
                  post_mod_loc,  # dim: 1 x G x D*K
                  post_mod_scale # dim: 1 x G x D*K
               ).to_event(1)
         )

         # Posterior distribution of 'base'.
         post_base_loc = pyro.param(
               "post_base_loc", # dim: 1 x G x N*B
               lambda: 1 * torch.ones(1,G,N*B).to(device, dtype)
         )
         post_base_scale = pyro.param(
               "post_base_scale", # dim: 1 x G x 1
               lambda: 3 * torch.ones(1,G,1).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )

         post_base = pyro.sample(
               name = "base",
               # dim: 1 x G | N*B
               fn = dist.Normal(
                  post_base_loc,  # dim: 1 x G x N*B
                  post_base_scale # dim: 1 x G x N*B
               ).to_event(1)
         )


if __name__ == "__main__":

   pyro.set_rng_seed(123)
   torch.manual_seed(123)

   K = int(sys.argv[1])
   in_fname = sys.argv[2]
   out_fname = sys.argv[3]

   # Optionally specify device through command line.
   device = "cuda" if len(sys.argv) == 4 else sys.argv[4]


   # Read in the data.
   data = new_sc_data(in_fname)

   cells, ctypes, batches, drugs, X = data
   ctypes = ctypes.to(device)
   batches = batches.to(device)
   drugs = drugs.to(device)
   X = X.to(device, torch.float64)
   mask = torch.ones_like(X).to(dtype=torch.bool)
   # HSPA8 and MT-ND4 show the strongest batch
   # effects when considering SAHA and PMA
   # treatments separately. To remove them from
   # the dataset, uncomment the lines below, which
   # just masks the entire columns for those genes.
   # Mask HSPA8 (idx 1236) and MT-ND4 (idx 4653).
   # mask[:,1236] = X[:,4653] = False

   # Mask HIV (last idx) in Jurkat cells only. They do
   # not contain HIV, so we count it as "unobserved".
#   mask[ctypes == 1,-1] = False
   #mask[ctypes < 2,-1] = False

   # Set the dimensions.
   B = int(batches.max() + 1)
   N = int(ctypes.max() + 1)
   D = int(drugs.max() + 1)
   G = int(X.shape[-1])

   data = batches, ctypes, drugs, X, mask
   # Use a warmup/decay learning-rate scheduler.
   scheduler = pyro.optim.PyroLRScheduler(
         scheduler_constructor = warmup_scheduler,
         optim_args = {
            "optimizer": torch.optim.AdamW,
            "optim_args": {"lr": 0.01}, "warmup": 400, "decay": 4000,
         },
         clip_args = {"clip_norm": 5.}
   )

   pyro.clear_param_store()

   ELBO = pyro.infer.Trace_ELBO if DEBUG else pyro.infer.JitTrace_ELBO
   svi = pyro.infer.SVI(
      model = model,
      guide = guide,
      optim = scheduler,
      loss = ELBO(
         num_particles = 16,
         vectorize_particles = True,
      )
   )

   loss = 0.
   for step in range(4000):
      loss += svi.step(data)
      scheduler.step()
      # Print progress on screen every 500 steps.
      if (step+1) % 500 == 0:
         sys.stderr.write(f"iter {step+1}: loss = {round(loss/1e9,3)}\n")
         loss = 0.

   # Model parameters.
   names = (
      "post_s_loc", "post_s_scale",
      "post_pi_0", "post_pi_1",
      "post_alpha", "post_theta_param",
      "post_base_loc", "post_base_scale",
      "post_mod_loc", "post_mod_scale",
      "post_shift_loc", "post_shift_scale",
   )
   ready = lambda x: x.detach().cpu().squeeze()
   params = { name: ready(pyro.param(name)) for name in names }

   # Posterior predictive sampling.
   predictive = pyro.infer.Predictive(
         model = model,
         guide = guide,
         num_samples = NUM_SAMPLES,
         return_sites = ("theta", "mod", "base", "_RETURN"),
   )
   with torch.no_grad():
      # Resample transcriptome (and "smoothed" estimates as well).
      sim = predictive(
            data = (batches, ctypes, drugs, None, mask),
            generate = X
      )

   smpl = {
         "tx": sim["_RETURN"][:,0,:,:].cpu(),
         "sm": sim["_RETURN"][:,1,:,:].cpu(),
   }

   # Save model and posterior predictive samples.
   torch.save({"params":params, "smpl":smpl}, out_fname)
