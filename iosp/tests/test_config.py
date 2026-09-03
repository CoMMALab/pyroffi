import jax.numpy as jnp

from iosp import config


def test_urdf_path_exists():
    assert config.URDF_PATH.exists(), f"URDF not found: {config.URDF_PATH}"


def test_srdf_path_exists():
    assert config.SRDF_PATH.exists(), f"SRDF not found: {config.SRDF_PATH}"


def test_theta_ik_star():
    assert config.THETA_IK_STAR.shape == (4,)
    assert config.THETA_IK_STAR.dtype == jnp.float32


def test_z_trajopt_star():
    assert config.Z_TRAJOPT_STAR.shape == (7,)
    assert config.Z_TRAJOPT_STAR.dtype == jnp.float32


def test_z_full_star():
    assert config.Z_FULL_STAR.shape == (4,)
    assert config.Z_FULL_STAR.dtype == jnp.float32


def test_q_start():
    assert config.Q_START.shape == (7,)
    assert config.Q_START.dtype == jnp.float32
