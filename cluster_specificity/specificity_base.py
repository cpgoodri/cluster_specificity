



def setup_loss_fn(params_default,
                  update_energy_fn,
                  create_test_structures_fn,
                  calculate_affinities_fn,
                  calculate_overlaps_fn,
                  objective_fn,
                  constraints_fn = None,
                  R_reference_structures = None,
                  R_motif = None):
  """ Set up a function to calculate the loss

  Args:
    paramd_default: dictionary of default energy parameters

    update_energy_fn: a function to update (or create) an energy function based
      on parameters. 

      signature:
      update_energy_fn(params) -> energy_fn

      where energy_fn takes a set of positions corresponding to a cluster
      and returns the scalar energys. 

    create_test_structures_fn: a function to create a list of test structures to
      be used by calculate_affinities_fn. These are starting points for 
      calculating the affinity of many different structures in different 
      orientations. 

      signature:
      create_test_structures_fn(key, R_reference_structures) -> R_test_init

      where R_test_init has shape(n_test, N, d). NOTE: if we want one "special"
      structure, i.e. the "correct" structure in the "correct" orientation and
      position, this needs to be handled here and then interpreted correctly in
      objective_fn. 

    calculate_affinities_fn: function to calculate the affinitiy of every 
      structure in R_test_init. This will usually include either an energy
      minimization or some sort of simulation.
      
      signature:
      calculate_affinities_fn(energy_fn, R_test_init) -> affinities,R_test_final

      where affinities has shape (n_test,) and R_test_final has shape 
      (n_test, N, d).

    calculate_overlaps_fn: function to calculate the overlaps of the final
      structures and R_motif.
      
      signature:
      calculate_overlap_fn(R_test_final, R_motif) -> overlaps

      where overlaps has shape (n_test,).

    objective_fn: function to calculate the objective based on affinities and 
      overlaps.

      signature:
      objective_fn(affinities, overlaps) -> objective

      where objective has shape (n_test,).

    constraints_fn: optional function to add constraints in the loss based only
      on the values of the parameters.

      signature:
      constraints_fn(params) -> constraints

      where constraints is a scalar.

    R_reference_structures: array of shape (n, N, d) of reference structures to 
      be used in create_test_structures_fn. 

    R_motif: array of shape (N_motif, d) giving the target motif

  Returns: objective + constraints

  """

  def loss_fn(theta, key):
    """ Calculate the loss

    Args:
      theta: pytree of parameter values. Must be recognizable by 
        update_energy_fn and constraints_fn
      key: a JAX prng key
    """
    params = {**params_default, **theta} 

    energy_fn = update_energy_fn(params)

    R_test_init = create_test_structures_fn(key, R_reference_structures)

    affinities, R_test_final = test_specificity(energy_fn, R_test_init)

    overlaps = calculate_overlaps_fn(R_test_final, R_motif)

    objective = objective_fn(affinities, overlaps)

    if constraints_fn is not None:
      constraints = constraints_fn(params)
    else:
      constraints = 0

    return objective + constraints
  
  return loss_fn
  













