import math
import os
from pathlib import Path

import torch


WANDB_PROJECT = "enrichment"
_API_KEY_NAMES = (
    "WANDB_API_KEY",
    "WANDB_API",
    "WANDB_KEY",
    "API_KEY",
    "API",
)
_KAGGLE_SECRET_NAMES = (
    "WANDB_API_KEY",
    "wandb_api_key",
    "WANDB_API",
    "wandb_api",
    "wandb",
)


def _find_env_file(start_dir=None):
    current = Path(start_dir or os.getcwd()).resolve()
    candidates = [current] + list(current.parents)
    for directory in candidates:
        env_path = directory / ".env"
        if env_path.is_file():
            return env_path
    return None


def _read_api_key_from_env_file(env_path):
    if env_path is None:
        return None
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    bare_values = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            bare_values.append(line.strip("'\""))
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key.upper() in _API_KEY_NAMES and value:
            return value

    if len(bare_values) == 1:
        return bare_values[0]
    return None


def _read_api_key_from_kaggle_secret():
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        return None

    try:
        client = UserSecretsClient()
    except Exception:
        return None

    for secret_name in _KAGGLE_SECRET_NAMES:
        try:
            value = client.get_secret(secret_name)
        except Exception:
            continue
        if value:
            return value
    return None


def load_wandb_api_key(start_dir=None):
    for key_name in _API_KEY_NAMES:
        value = os.environ.get(key_name)
        if value:
            os.environ["WANDB_API_KEY"] = value
            return value

    value = _read_api_key_from_env_file(_find_env_file(start_dir))
    if value:
        os.environ["WANDB_API_KEY"] = value
        return value

    value = _read_api_key_from_kaggle_secret()
    if value:
        os.environ["WANDB_API_KEY"] = value
        return value
    return None


def _config_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_wandb_config(args, run_name, output_dir):
    config = {key: _config_value(value) for key, value in vars(args).items()}
    config.update({
        "wandb_project": WANDB_PROJECT,
        "wandb_run_name": run_name,
        "output_dir": output_dir,
    })
    return config


def init_wandb(args, run_name, output_dir, logger=None):
    if not getattr(args, "use_wandb", True):
        return None

    api_key = load_wandb_api_key(start_dir=os.getcwd())
    if not api_key:
        if logger is not None:
            logger.warning("W&B disabled: no API key found in environment, .env, or Kaggle secrets.")
        return None

    try:
        import wandb
    except ImportError:
        if logger is not None:
            logger.warning("W&B disabled: wandb package is not installed.")
        return None

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    os.environ.setdefault("WANDB_NAME", run_name)

    try:
        wandb.login(key=api_key, relogin=False)
        init_kwargs = {
            "project": WANDB_PROJECT,
            "name": run_name,
            "config": build_wandb_config(args, run_name, output_dir),
            "dir": output_dir,
            "resume": "allow",
        }
        try:
            run = wandb.init(
                **init_kwargs,
                settings=wandb.Settings(start_method="thread"),
            )
        except TypeError:
            run = wandb.init(**init_kwargs)
        wandb.define_metric("global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("train_epoch/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        run.summary["output_dir"] = output_dir
        return run
    except Exception as error:
        if logger is not None:
            logger.warning("W&B disabled after initialization failure: {}".format(error))
        return None


def _scalar_for_wandb(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().float().item()
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return None


def log_wandb(run, metrics, step=None, epoch=None, prefix=None):
    if run is None or not metrics:
        return

    payload = {}
    for key, value in metrics.items():
        scalar = _scalar_for_wandb(value)
        if scalar is None:
            continue
        metric_key = "{}/{}".format(prefix, key) if prefix else key
        payload[metric_key] = scalar

    if step is not None:
        payload["global_step"] = int(step)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if payload:
        run.log(payload)


def finish_wandb(run):
    if run is not None:
        run.finish()
