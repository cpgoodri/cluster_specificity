
import jax.numpy as jnp
from jax import vmap
from jax_md import space, energy, smap



reference_default_params = \
    {
        'R_net':      jnp.zeros( (100,3), dtype=jnp.float64),
        'R_patches':  jnp.zeros( (2,3), dtype=jnp.float64),

        'colloid_diameter':        1.0,
        'cluster_cluster_epsilon': 1.0,  #this is probably always going to be our energy scale
        'cluster_cluster_alpha':   5.0,

        'net_particle_diameter': 0.4,
        'cluster_net_epsilon':   100.0,
        
        'patch_diameter':        0.,
        'cluster_patch_epsilon': 1.0,
        'cluster_patch_alpha':   5.0
    }



def morse_pair_nocutoff( displacement_or_metric,
                         sigma=1.0,
                         epsilon=1.0,
                         alpha=5.0,
                         per_particle=False):
  """Convenience wrapper to compute :ref:`Morse energy <morse-pot>` over a system."""
  return smap.pair(
    energy.morse,
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    ignore_unused_parameters=True,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha,
    reduce_axis=(1,) if per_particle else None)

def get_bonds_from_species(species, i, j):
  """ Given a species list, return an array of shape (n,2) corresponding to all
        the bonds between species i and j

      We are hard-coding the definition of the species: 
        0 is the cluster, 
        1 is the net, 
        2 is the patches
  """
  get_bond_ids = vmap(vmap(lambda a,b: jnp.array([a,b]), in_axes=(0,None)), in_axes=(None,0))
  return get_bond_ids(jnp.where(species==i),jnp.where(species==j)).reshape(-1,2)


def setup_cluster_energy(species,

                         ):

  displacement, shift = space.free()

  cluster_energy_base = morse_pair_nocutoff(displacement)

  net_cluster_bonds = get_bonds_from_species(species, 0, 1)
  cluster_net_energy_base = smap.bond(energy.soft_sphere,
                                      space.canonicalize_displacement_or_metric(displacement),
                                      net_cluster_bonds
                                      )

  cluster_patch_bonds = get_bonds_from_species(species, 0, 2)
  cluster_patch_energy_base = smap.bond( morse_pair_nocutoff,
                                         space.canonicalize_displacement_or_metric(displacement),
                                         cluster_patch_bonds,
                                        )


  def update_energy_fn(params):
    def energy_fn(R_cluster, **unused_kwargs):
      #we are not checking that the R_cluster, params['R_net'], and 
      # params['R_patches'] have the shapes that correspond to speceies 
      R = jnp.concatenate((R_cluster, params['R_net'], params['R_patches']), axis=0)

      cluster_energy = cluster_energy_base(R_cluster, 
                                           sigma = params['colloid_diameter'],
                                           epsilon = params['cluster_cluster_epsilon'],
                                           alpha = params['cluster_cluster_alpha']
                                           )

      cluster_net_sigma = (params['colloid_diameter'] + params['net_particle_diameter']) / 2,
      cluster_net_energy = cluster_net_energy_base(R,
                                                   sigma = cluster_net_sigma,
                                                   epsilon = params['cluster_net_epsilon']
                                                   )

      cluster_patch_sigma = (params['patch_diameter'] + params['colloid_diameter']) / 2
      cluster_patch_energy = cluster_patch_energy_base(R,
                                                       sigma = cluster_patch_sigma,
                                                       epsilon = params['cluster_patch_epsilon'],
                                                       alpha = params['cluster_patch_alpha']
                                                      )


      return cluster_energy + cluster_net_energy + cluster_patch_energy

    return energy_fn

  return update_energy_fn




