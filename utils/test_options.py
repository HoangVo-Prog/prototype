import argparse


DATASET_CHOICES = ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid")


def _add_boolean_override(parser, name, help_text, negative_help=None):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--{}".format(name),
        dest=name,
        action="store_true",
        default=None,
        help=help_text,
    )
    group.add_argument(
        "--no_{}".format(name),
        dest=name,
        action="store_false",
        default=None,
        help=negative_help or "disable {}".format(name.replace("_", " ")),
    )


def get_test_args():
    parser = argparse.ArgumentParser(
        description="ITSELF inference and cross-domain evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run_dir",
        default=None,
        help="training run directory containing configs.yaml and best.pth",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        help="training config file; defaults to <run_dir>/configs.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path; defaults to <run_dir>/best.pth",
    )
    parser.add_argument(
        "--source_domain",
        default=None,
        choices=DATASET_CHOICES,
        help="dataset used to train the checkpoint; defaults to dataset_name in config",
    )
    parser.add_argument(
        "--target_domain",
        default=None,
        choices=DATASET_CHOICES,
        help="single target domain alias for --target_domains",
    )
    parser.add_argument(
        "--target_domains",
        nargs="+",
        default=None,
        choices=DATASET_CHOICES,
        help="one or more target datasets to evaluate",
    )
    parser.add_argument(
        "--cross_domain",
        action="store_true",
        help="mark the run as cross-domain evaluation in logs/results",
    )

    parser.add_argument("--root_dir", default=None, help="dataset root override")
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="device for inference, for example auto, cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="sets CUDA_VISIBLE_DEVICES before torch/model imports",
    )

    parser.add_argument(
        "--output_eval_dir",
        default=None,
        help="directory for test_log.txt and optional JSON results",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="save collected metrics to eval_results.json",
    )
    parser.add_argument(
        "--results_file",
        default=None,
        help="explicit JSON output path; also enables saving",
    )
    parser.add_argument(
        "--i2t_metric",
        action="store_true",
        help="also report image-to-text retrieval metrics",
    )

    parser.add_argument("--seed", type=int, default=None)
    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=None,
        help="force deterministic PyTorch/CUDA settings",
    )
    deterministic_group.add_argument(
        "--non_deterministic",
        dest="deterministic",
        action="store_false",
        default=None,
        help="disable strict deterministic PyTorch/CUDA settings",
    )
    parser.add_argument(
        "--deterministic_warn_only",
        action="store_true",
        default=None,
        help="warn instead of raising on nondeterministic PyTorch operations",
    )

    _add_boolean_override(
        parser,
        "only_global",
        "use only global features during evaluation",
        "use global plus GRAB features during evaluation",
    )
    _add_boolean_override(
        parser,
        "target_enrichment",
        "enable target-aware text enrichment during evaluation",
        "disable target-aware text enrichment during evaluation",
    )
    parser.add_argument(
        "--enrichment_space",
        choices=("global", "grab"),
        default=None,
        help="target enrichment retrieval space override",
    )

    args = parser.parse_args()
    if args.run_dir is None and args.config_file is None:
        parser.error("provide --run_dir or --config_file")
    if args.target_domain is not None and args.target_domains is not None:
        parser.error("use either --target_domain or --target_domains, not both")
    if args.target_domain is not None:
        args.target_domains = [args.target_domain]
    return args
