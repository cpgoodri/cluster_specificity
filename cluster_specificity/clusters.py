import jax.numpy as jnp
from jax_transformations3d import jax_transformations3d as jts
from jax_md import space



def com(R):
  return jnp.mean(R, axis=0)


def get_reference_clusters_N6()
  displacement, shift = space.free()

  # coordinates are for diameter=1 particles
  Rpoly = jnp.array([[0.74243083, -0.0353437 ,  0.        ],
                     [-0.8670022 , -0.3114399 ,  0.        ],
                     [ 0.3114399 ,  0.8670022 ,  0.        ],
                     [ 0.0353437 , -0.74243083,  0.        ],
                     [-0.11110611,  0.11110611,  0.50000271],
                     [-0.11110611,  0.11110611, -0.50000271]], dtype=jnp.float64)
  Rpoly = Rpoly - com(Rpoly)

  Rocta = jnp.array([[0.70710678118654752440, 0., 0.],
                     [-0.70710678118654752440, 0., 0.],
                     [0., 0.70710678118654752440, 0.],
                     [0., -0.70710678118654752440, 0.],
                     [0., 0., 0.70710678118654752440],
                     [0., 0., -0.70710678118654752440]], dtype=jnp.float64)
  Rocta = Rocta - com(Rocta)

  Rpoly = Rpoly[jnp.array([1,2,4,5,0,3])]
  Rocta = Rocta[jnp.array([1,2,4,5,0,3])]

  v0 = displacement(Rpoly[0],Rpoly[1])
  v1 = displacement(Rpoly[2],Rpoly[3])
  v0 = v0 / jnp.linalg.norm(v0)
  v1 = v1 / jnp.linalg.norm(v1)
  v2 = jnp.cross(v0,v1)

  v = jnp.array([v0, v1, v2])
  v_target = jnp.identity(3)
  M = jts.affine_matrix_from_points(v.T,v_target.T, shear=False, scale=False)
  Rpoly = jts.matrix_apply(M, Rpoly)
  Rocta = jts.matrix_apply(M, Rocta)

  return jnp.array([Rpoly, Rocta])




def get_reference_clusters(N):
  if N == 6:
    return get_reference_cluster_N6()
  else:
    assert False






