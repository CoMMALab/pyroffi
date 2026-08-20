"""Small port-local utilities (jax cache config etc.)."""


def jax_cache_on():
    import jax
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_spasm_pyroffi")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
