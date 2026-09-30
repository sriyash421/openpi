# Source before any openpi command: every cache, venv, checkpoint and dataset lives in scrubbed.
export S=/gscratch/scrubbed/sriyash/openpi_pt OPENPI=/gscratch/socialrl/sriyash/openpi_pt
export UV_PROJECT_ENVIRONMENT=$S/venv UV_CACHE_DIR=$S/uv_cache UV_PYTHON_INSTALL_DIR=$S/uv_python UV_LINK_MODE=copy
export OPENPI_DATA_HOME=$S/openpi_data HF_HOME=$S/hf HF_LEROBOT_HOME=$S/hf/lerobot
export XDG_CACHE_HOME=$S/xdg_cache TMPDIR=$S/tmp WANDB_DIR=$S/wandb TORCHINDUCTOR_CACHE_DIR=$S/xdg_cache/inductor
export LIBERO_CONFIG_PATH=$S/libero_cfg
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=$S/egl/10_nvidia.json
export GIT_LFS_SKIP_SMUDGE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=$S/xdg_cache/numba PYTHONDONTWRITEBYTECODE=1
# Persistent XLA compilation cache: training and servers restart every on-policy iteration; don't recompile each time.
export JAX_COMPILATION_CACHE_DIR=$S/xdg_cache/jax JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
