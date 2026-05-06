import logging
import os
from pathlib import Path

from utils.comm import get_rank


class ExperimentTracker:
    def __init__(self, args, tb_writer=None):
        self.args = args
        self.tb_writer = tb_writer
        self.logger = logging.getLogger("ITSELF.tracker")
        self.enabled = False
        self.run = None

        if not getattr(args, "wandb", False) or get_rank() != 0:
            return

        try:
            import wandb
        except ImportError:
            self.logger.warning("W&B requested but wandb is not installed; disabling W&B logging.")
            return

        self._login_wandb(wandb)

        tags = getattr(args, "wandb_tags", None) or []
        self.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            tags=tags,
            mode=args.wandb_mode,
            dir=args.output_dir,
            config=vars(args),
        )
        self.enabled = self.run is not None

    def _login_wandb(self, wandb):
        api_key = (
            os.environ.get("WANDB_API_KEY")
            or self._read_wandb_key_from_env_file()
            or self._read_wandb_key_from_kaggle_secrets()
        )
        if not api_key:
            return

        os.environ["WANDB_API_KEY"] = api_key
        try:
            wandb.login(key=api_key, relogin=False)
        except Exception as exc:
            self.logger.warning("W&B login failed during automatic credential loading: %s", exc)

    def _read_wandb_key_from_env_file(self):
        env_candidates = [
            Path.cwd() / ".env",
            Path(getattr(self.args, "output_dir", "")).resolve().parent / ".env"
            if getattr(self.args, "output_dir", None)
            else None,
        ]
        for env_path in env_candidates:
            if env_path is None or not env_path.is_file():
                continue
            api_key = self._extract_env_value(env_path, "WANDB_API_KEY")
            if api_key:
                self.logger.info("Loaded W&B API key from %s", env_path)
                return api_key
        return None

    def _read_wandb_key_from_kaggle_secrets(self):
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is None:
            return None

        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError:
            return None

        client = UserSecretsClient()
        for secret_name in ("WANDB_API_KEY", "wandb_api_key", "wandb-key"):
            try:
                api_key = client.get_secret(secret_name)
            except Exception:
                continue
            if api_key:
                self.logger.info("Loaded W&B API key from Kaggle secret '%s'", secret_name)
                return api_key
        return None

    @staticmethod
    def _extract_env_value(env_path, key):
        prefix = f"{key}="
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or not stripped.startswith(prefix):
                continue
            value = stripped[len(prefix):].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            return value
        return None

    @staticmethod
    def _as_scalar(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
            if value.numel() == 1:
                return float(value.item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def log_scalars(self, metrics, step=None):
        if not self.enabled:
            return

        payload = {}
        for name, value in metrics.items():
            scalar = self._as_scalar(value)
            if scalar is not None:
                payload[name] = scalar
        if payload:
            self.run.log(payload, step=step)

    def add_scalar(self, name, value, step):
        if self.tb_writer is not None:
            scalar = self._as_scalar(value)
            if scalar is not None:
                self.tb_writer.add_scalar(name, scalar, step)

    def add_scalars(self, metrics, step):
        for name, value in metrics.items():
            self.add_scalar(name, value, step)
        self.log_scalars(metrics, step=step)

    def update_summary(self, metrics):
        if not self.enabled:
            return

        for name, value in metrics.items():
            scalar = self._as_scalar(value)
            if scalar is not None:
                self.run.summary[name] = scalar

    def finish(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.enabled and self.run is not None:
            self.run.finish()
