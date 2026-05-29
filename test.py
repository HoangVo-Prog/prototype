import os
import os.path as op

from utils.test_options import get_test_args


def _resolve_user_path(path, base_dir=None):
    if path is None:
        return None
    path = op.expanduser(path)
    if op.isabs(path):
        return op.abspath(path)
    if base_dir is not None:
        return op.abspath(op.join(base_dir, path))
    return op.abspath(path)


def _resolve_paths(cli_args):
    run_dir = _resolve_user_path(cli_args.run_dir)
    config_file = _resolve_user_path(cli_args.config_file, run_dir)
    if config_file is None:
        config_file = op.join(run_dir, "configs.yaml")

    checkpoint_base_dir = run_dir if run_dir is not None else op.dirname(config_file)
    checkpoint = _resolve_user_path(cli_args.checkpoint, checkpoint_base_dir)
    if checkpoint is None:
        checkpoint = op.join(checkpoint_base_dir, "best.pth")

    eval_dir = _resolve_user_path(cli_args.output_eval_dir, checkpoint_base_dir)
    if eval_dir is None:
        eval_dir = op.join(checkpoint_base_dir, "eval")

    results_file = _resolve_user_path(cli_args.results_file, eval_dir)
    if results_file is None:
        results_file = op.join(eval_dir, "eval_results.json")

    return config_file, checkpoint, eval_dir, results_file


def _set_if_not_present(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _ensure_inference_defaults(args, config_file):
    config_dir = op.dirname(config_file)
    _set_if_not_present(args, "output_dir", config_dir)
    _set_if_not_present(args, "root_dir", "data")
    _set_if_not_present(args, "num_workers", 4)
    _set_if_not_present(args, "test_batch_size", 512)
    _set_if_not_present(args, "seed", 1)
    _set_if_not_present(args, "deterministic", True)
    _set_if_not_present(args, "deterministic_warn_only", False)
    _set_if_not_present(args, "training", False)
    _set_if_not_present(args, "distributed", False)
    _set_if_not_present(args, "local_rank", 0)
    _set_if_not_present(args, "only_global", False)
    _set_if_not_present(args, "target_enrichment", False)
    _set_if_not_present(args, "enrichment_space", "global")


def _apply_cli_overrides(args, cli_args, config_file, checkpoint, eval_dir):
    args.training = False
    args.distributed = False
    args.config_file = config_file
    args.checkpoint = checkpoint
    args.eval_output_dir = eval_dir

    for name in (
        "root_dir",
        "test_batch_size",
        "num_workers",
        "seed",
        "deterministic",
        "deterministic_warn_only",
        "only_global",
        "target_enrichment",
        "enrichment_space",
    ):
        value = getattr(cli_args, name, None)
        if value is not None:
            setattr(args, name, value)


def _task_enabled(args, task_name):
    loss_names = str(getattr(args, "loss_names", ""))
    return task_name in [name.strip() for name in loss_names.split("+")]


def _infer_num_classes_from_checkpoint(checkpoint_file, args, logger):
    if not _task_enabled(args, "cid"):
        return None

    import torch
    from utils.checkpoint import unwrap_checkpoint_state_dict

    try:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        state_dict = unwrap_checkpoint_state_dict(checkpoint)
    except Exception as error:
        logger.warning(
            "Could not infer source num_classes from checkpoint: {}".format(error)
        )
        return None

    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key == "classifier_global.weight" and value.ndim == 2:
            num_classes = int(value.shape[0]) - 1
            if num_classes > 0:
                logger.info(
                    "Inferred source num_classes={} from checkpoint classifier".format(
                        num_classes
                    )
                )
                return num_classes

    logger.warning("Checkpoint does not contain classifier_global.weight")
    return None


def _num_classes_from_source_dataset(args, source_domain, logger):
    from datasets.build import __factory as dataset_factory

    if source_domain not in dataset_factory:
        raise ValueError("Unsupported source domain: {}".format(source_domain))

    dataset = dataset_factory[source_domain](root=args.root_dir, verbose=False)
    num_classes = len(dataset.train_id_container)
    logger.info(
        "Read source num_classes={} from {} train split".format(
            num_classes,
            source_domain,
        )
    )
    return num_classes


def _resolve_num_classes(args, source_domain, checkpoint_file, logger):
    num_classes = _infer_num_classes_from_checkpoint(checkpoint_file, args, logger)
    if num_classes is not None:
        return num_classes
    if _task_enabled(args, "cid"):
        return _num_classes_from_source_dataset(args, source_domain, logger)
    return 0


def _resolve_device(device_name):
    import torch

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def _build_model(args, source_domain, checkpoint_file, device, logger):
    from model import build_model
    from utils.checkpoint import Checkpointer

    num_classes = _resolve_num_classes(args, source_domain, checkpoint_file, logger)
    model = build_model(args, num_classes)
    checkpointer = Checkpointer(model, logger=logger)
    checkpointer.load(f=checkpoint_file)
    model.to(device)
    model.eval()
    return model


def _target_domains(cli_args, source_domain):
    if cli_args.target_domains:
        return list(cli_args.target_domains)
    return [source_domain]


def _source_check_enabled(cli_args):
    if cli_args.source_check is None:
        return True
    return bool(cli_args.source_check)


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _evaluate_domain(model, args, target_domain, cli_args, logger):
    from datasets import build_dataloader
    from utils.metrics import Evaluator

    logger.info("Evaluating target domain: {}".format(target_domain))
    args.dataset_name = target_domain
    args.training = False

    test_img_loader, test_txt_loader, _ = build_dataloader(args)
    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    top1 = evaluator.eval(
        model.eval(),
        i2t_metric=cli_args.i2t_metric,
        use_target_enrichment=getattr(args, "target_enrichment", False),
    )

    metrics = {
        key: _json_safe(value)
        for key, value in getattr(evaluator, "last_metrics", {}).items()
    }
    metrics["best_R1"] = float(top1)
    metrics["best_task"] = getattr(evaluator, "last_best_task", None)
    return metrics


def main():
    cli_args = get_test_args()
    if cli_args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.cuda_visible_devices

    from utils.iotools import load_train_configs, write_json
    from utils.reproducibility import configure_reproducibility

    config_file, checkpoint, eval_dir, results_file = _resolve_paths(cli_args)
    if not op.isfile(config_file):
        raise FileNotFoundError("Config file not found: {}".format(config_file))
    if not op.isfile(checkpoint):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))

    args = load_train_configs(config_file)
    _ensure_inference_defaults(args, config_file)
    _apply_cli_overrides(args, cli_args, config_file, checkpoint, eval_dir)

    configure_reproducibility(
        args.seed,
        deterministic=args.deterministic,
        warn_only=args.deterministic_warn_only,
    )

    from utils.logger import setup_logger

    logger = setup_logger("ITSELF", save_dir=eval_dir, if_train=False)
    source_domain = cli_args.source_domain or args.dataset_name
    targets = _target_domains(cli_args, source_domain)
    cross_domain = cli_args.cross_domain or any(domain != source_domain for domain in targets)
    run_source_check = _source_check_enabled(cli_args)

    logger.info("Config file: {}".format(config_file))
    logger.info("Checkpoint: {}".format(checkpoint))
    logger.info("Source domain: {}".format(source_domain))
    logger.info("Target domains: {}".format(", ".join(targets)))
    logger.info("Cross-domain evaluation: {}".format(cross_domain))
    logger.info("Source-domain sanity inference: {}".format(run_source_check))
    logger.info("Evaluation output: {}".format(eval_dir))

    device = _resolve_device(cli_args.device)
    logger.info("Device: {}".format(device))
    model = _build_model(args, source_domain, checkpoint, device, logger)

    results = {
        "config_file": config_file,
        "checkpoint": checkpoint,
        "source_domain": source_domain,
        "target_domains": targets,
        "cross_domain": cross_domain,
        "source_check_enabled": run_source_check,
        "source_check": None,
        "device": str(device),
        "results": {},
    }

    source_check_metrics = None
    if run_source_check:
        logger.info(
            "Running source-domain sanity inference before target evaluation: {}".format(
                source_domain
            )
        )
        source_check_metrics = _evaluate_domain(
            model,
            args,
            source_domain,
            cli_args,
            logger,
        )
        results["source_check"] = {
            "domain": source_domain,
            "metrics": source_check_metrics,
        }
    else:
        logger.info("Skipping source-domain sanity inference")

    for target_domain in targets:
        if run_source_check and target_domain == source_domain:
            logger.info(
                "Reusing source-domain sanity inference for target domain: {}".format(
                    target_domain
                )
            )
            results["results"][target_domain] = source_check_metrics
            continue
        results["results"][target_domain] = _evaluate_domain(
            model,
            args,
            target_domain,
            cli_args,
            logger,
        )

    if cli_args.save_json or cli_args.results_file is not None:
        write_json(results, results_file)
        logger.info("Saved evaluation results to {}".format(results_file))


if __name__ == "__main__":
    main()
