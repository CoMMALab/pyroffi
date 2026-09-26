"""The shared kernel builder caches by content and never leaves a half-built library behind."""

import shutil

import pytest

if shutil.which("nvcc") is None:
    pytest.skip("needs nvcc", allow_module_level=True)

from pyroffi.cuda_kernels._traced import build_shared_library

FLAGS = ["-O0", "--shared", "--compiler-options", "-fPIC"]


def test_cache_hit_and_failed_build_cleanup(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("PYROFFI_TRACED_CACHE", str(cache))
    src = tmp_path / "k.cu"
    src.write_text('#include "gen.cuh"\nextern "C" int value() { return VALUE; }\n')

    first = build_shared_library("k", src, {"gen.cuh": "#define VALUE 1\n"}, FLAGS)
    assert first.is_file()
    assert build_shared_library("k", src, {"gen.cuh": "#define VALUE 1\n"}, FLAGS) == first
    other = build_shared_library("k", src, {"gen.cuh": "#define VALUE 2\n"}, FLAGS)
    assert other != first  # generated header is part of the key

    with pytest.raises(RuntimeError, match="nvcc failed"):
        build_shared_library("k", src, {"gen.cuh": "#define VALUE (\n"}, FLAGS)
    assert sorted(p.name for p in cache.iterdir()) == sorted([first.parent.name, other.parent.name])
