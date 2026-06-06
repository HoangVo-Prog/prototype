import argparse


_EXTRACTOR_BASE_MODES = (
    "global",
    "horizontal",
    "vertical",
    "grid",
    "retrieval_backbone",
    "cluster",
    "cluster_residual",
    "cluster_density",
    "cluster_rarity",
)
_LEGACY_EXTRACTOR_MODE_ALIASES = {
    "global_horizontal": "global,horizontal",
    "global_vertical": "global,vertical",
    "global_grid": "global,grid",
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_extractor_mode(value):
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("--extractor_mode must be a comma-separated string")

    raw_tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not raw_tokens:
        raise argparse.ArgumentTypeError("--extractor_mode must contain at least one mode")

    expanded_tokens = []
    for token in raw_tokens:
        alias = _LEGACY_EXTRACTOR_MODE_ALIASES.get(token)
        if alias is not None:
            expanded_tokens.extend(alias.split(","))
        else:
            expanded_tokens.append(token)

    normalized_tokens = []
    seen = set()
    for token in expanded_tokens:
        if token not in _EXTRACTOR_BASE_MODES:
            raise argparse.ArgumentTypeError(
                f"--extractor_mode supports comma-separated values from {_EXTRACTOR_BASE_MODES}; got '{value}'"
            )
        if token not in seen:
            seen.add(token)
            normalized_tokens.append(token)
    return ",".join(normalized_tokens)


def get_args():
    parser = argparse.ArgumentParser(description="ITSELF Args")
    parser.add_argument("--tau", default=0.015, type=float)
    parser.add_argument("--select_ratio", default=0.4, type=float)
    parser.add_argument("--margin", default=0.1, type=float)
    parser.add_argument("--lambda1_weight", default=0.5, type=float)
    parser.add_argument("--lambda2_weight", default=3.5, type=float)

    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--output_dir", default="run_logs")
    parser.add_argument("--name", default="ITSELF", help="experiment name to save")
    parser.add_argument("--seed", default=1, type=int,
                        help="base seed for model init, samplers, and dataloader workers")
    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument("--deterministic", dest="deterministic", action="store_true",
                                     help="enable deterministic PyTorch/CUDA settings")
    deterministic_group.add_argument("--non_deterministic", dest="deterministic", action="store_false",
                                     help="disable strict deterministic PyTorch/CUDA settings")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--deterministic_warn_only", action="store_true", default=False,
                        help="warn instead of raising when PyTorch encounters a nondeterministic operation")
    parser.add_argument("--log_period", default=20)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--finetune_clip", type=str, default="",
                        help="load compatible ITSELF weights (including CLIP/GRAB modules) from a CLIP or ITSELF checkpoint")
    parser.add_argument("--freeze_host", action="store_true", default=False,
                        help="freeze host CLIP/ITSELF parameters and train only target enrichment modules")
    parser.add_argument("--pretrain", type=str, default="")


    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')
    parser.add_argument("--txt_aug", default=True, action='store_true')
    
    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='tal+cid', help="which loss to use ['cid, tal']")
    host_loss_group = parser.add_mutually_exclusive_group()
    host_loss_group.add_argument("--use_host_loss", dest="use_host_loss", action="store_true",
                                 help="enable host cid/tal loss in the optimized objective")
    host_loss_group.add_argument("--no_use_host_loss", dest="use_host_loss", action="store_false",
                                 help="disable host cid/tal loss in the optimized objective")
    parser.set_defaults(use_host_loss=True)
    parser.add_argument("--lambda_host", type=float, default=1.0,
                        help="weight applied to the combined host loss")

    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--lr_total_epochs", type=int, default=-1,
                        help="epoch scale for LR scheduler; -1 follows num_epoch")
    parser.add_argument("--milestones", type=int, nargs='+', default=(45, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="identity", help="choose sampler from [identity, random]")
    parser.add_argument("--num_instance", type=int, default=2)
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test", dest='training', default=True, action='store_false')
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--use_wandb", dest="use_wandb", action="store_true",
                             help="enable automatic Weights & Biases logging")
    wandb_group.add_argument("--no_wandb", dest="use_wandb", action="store_false",
                             help="disable Weights & Biases logging")
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--wandb_project", default="enrichment",
                        help="Weights & Biases project name")

    ### GRAB
    parser.add_argument("--only_global", action='store_true')
    parser.add_argument("--return_all", action='store_true')
    parser.add_argument("--topk_type", type=str, default='mean', help='[mean, std, custom, layer_index]')
    parser.add_argument("--layer_index", type=int, default=-1, help='which layer attention to use: [0, 11]')
    parser.add_argument("--average_attn_weights", type=bool, default=True)
    parser.add_argument("--modify_k", action='store_true')

    ######################## target-aware text enrichment ########################
    parser.add_argument("--target_enrichment", action='store_true',
                        help="enable target-aware prototype text enrichment")
    parser.add_argument("--enrichment_start", type=int, default=1,
                        help="first training epoch that enables target enrichment; earlier epochs train host losses only")
    parser.add_argument("--enrichment_space", type=str, default="global",
                        choices=["global", "grab"],
                        help="feature space to enrich when target_enrichment is enabled")
    parser.add_argument("--top_m", type=int, default=32,
                        help="number of host-ranked local images used for enrichment")
    parser.add_argument("--topm_rank_space", type=str, default="host_global",
                        choices=["host_global", "retrieval", "hybrid_global_grab"],
                        help="feature space used to select top-M target images for enrichment")
    parser.add_argument("--topm_rank_lambda", type=float, default=0.5,
                        help="global score weight for --topm_rank_space hybrid_global_grab: "
                             "lambda*global + (1-lambda)*grab")
    parser.add_argument("--extractor_mode", type=parse_extractor_mode, default="global,horizontal",
                        help="comma-separated evidence providers")
    parser.add_argument("--num_parts", type=int, default=6,
                        help="number of partitions for horizontal/vertical extractors; grid uses num_parts x num_parts") 
    parser.add_argument("--target_relative_space", type=str, default="host_global",
                        choices=["host_global", "retrieval"],
                        help="target-pool feature space used for target-relative evidence providers")
    parser.add_argument("--target_relative_num_clusters", type=int, default=16,
                        help="number of label-free clusters for target-relative evidence providers")
    parser.add_argument("--target_relative_cluster_method", type=str, default="kmeans",
                        choices=["kmeans"],
                        help="label-free clustering method for target-relative evidence providers")
    parser.add_argument("--evidence_token_budget", type=int, default=0,
                        help="0 disables the evidence-token budget; positive values validate extractor token count")
    parser.add_argument("--evidence_projection", type=str, default="auto",
                        choices=["auto", "linear", "none"],
                        help="projection policy for evidence providers whose source dim differs from the shared dim")
    parser.add_argument("--use_freeze_indices", "--freeze_indices",
                        dest="use_freeze_indices", action="store_true", default=False,
                        help="precompute frozen host top-M rankings once and reuse them for top-M selection")
    parser.add_argument("--pnp_text_only", action="store_true", default=False,
                        help="for frozen plug-and-play global training, encode only batch text and use frozen target cache for images")
    parser.add_argument("--robust_hard_k", "--hard_neg_k", dest="robust_hard_k",
                        type=int, default=32,
                        help="number of raw-score hard negatives R used by robust margin loss")
    parser.add_argument("--enrich_gamma", type=float, default=None,
                        help="static residual strength; valid only with --residual_gate static")
    parser.add_argument("--residual_gate", "--gate_mode", dest="residual_gate",
                        type=str, default="residual", choices=["static", "residual"],
                        help="static uses --enrich_gamma; residual learns a per-query residual gate")
    parser.add_argument("--residual_gate_hidden_dim", type=int, default=128,
                        help="hidden dimension for the learned residual gate MLP")
    parser.add_argument("--recompute_level", type=str, default="epoch",
                        choices=["epoch", "step"],
                        help="unit used by recompute_interval for target-pool refresh")
    parser.add_argument("--recompute_interval", type=int, default=1,
                        help="-1 computes the target pool once; otherwise refresh every N epochs or steps")
    parser.add_argument("--pool_interval", dest="recompute_interval", type=int,
                        default=argparse.SUPPRESS,
                        help="alias for --recompute_interval")
    
    ######################## mlp-mixer module settings ########################
    parser.add_argument("--context_module", type=str, default="mixer",
                        choices=["mixer"],
                        help="context construction module for target enrichment")
    parser.add_argument("--mixer_dim", type=int, default=256,
                        help="rank-part mixer bottleneck dimension")
    parser.add_argument("--mixer_depth", type=int, default=2,
                        help="number of stacked rank-part mixer blocks")
    parser.add_argument("--mixer_hidden_part", type=int, default=32,
                        help="hidden dimension for prototype-slot mixing")
    parser.add_argument("--mixer_hidden_rank", type=int, default=64,
                        help="hidden dimension for host-rank mixing")
    parser.add_argument("--mixer_hidden_channel", type=int, default=512,
                        help="hidden dimension for mixer-channel mixing")
    parser.add_argument("--mixer_hidden_readout", type=int, default=128,
                        help="hidden dimension for MLP or hybrid token readout")
    parser.add_argument("--context_pooling", "--mixer_context_pooling",
                        dest="context_pooling", type=str, default="mlp",
                        choices=["mlp", "late_attention", "hybrid_attention"],
                        help="context pooling after rank-part mixing")    
    
    ######################## target-aware loss settings ########################
    parser.add_argument("--lambda_ret", type=float, default=1.0,
                        help="weight for the target-pool retrieval loss")
    parser.add_argument("--lambda_rob", type=float, default=0.1,
                        help="weight for robust no-harm/margin-gain loss")
    parser.add_argument("--lambda_gain", type=float, default=1.0,
                        help="weight for the margin-gain term inside the robust loss")
    parser.add_argument("--gain_margin", type=float, default=0.01,
                        help="required enriched-vs-raw retrieval margin gain")
    parser.add_argument("--use_target_retrieval_loss", action="store_true", default=False,
                        help="enable the primary target-pool retrieval loss")
    parser.add_argument("--use_target_robust_loss", action="store_true", default=False,
                        help="enable the robust no-harm and margin-gain loss")
    
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be in [0, 2**32)")
    args.wandb_project = args.wandb_project.strip()
    if not args.wandb_project:
        parser.error("--wandb_project must not be empty")
    if args.enrichment_start < 1:
        parser.error("--enrichment_start must be a positive integer")
    if args.freeze_host and not args.target_enrichment:
        parser.error("--freeze_host requires --target_enrichment")
    if args.use_freeze_indices and not args.target_enrichment:
        parser.error("--use_freeze_indices requires --target_enrichment")
    if args.topm_rank_lambda < 0.0 or args.topm_rank_lambda > 1.0:
        parser.error("--topm_rank_lambda must be in [0, 1]")
    if args.target_relative_num_clusters < 1:
        parser.error("--target_relative_num_clusters must be a positive integer")
    if args.evidence_token_budget < 0:
        parser.error("--evidence_token_budget must be non-negative")
    if args.evidence_token_budget > 0:
        from model.enrichment.prototypes import prototype_slot_count
        if prototype_slot_count(args.extractor_mode, args.num_parts) > args.evidence_token_budget:
            parser.error("--extractor_mode exceeds --evidence_token_budget")
    if (
        args.target_enrichment
        and args.topm_rank_space == "hybrid_global_grab"
        and args.only_global
    ):
        parser.error("--topm_rank_space hybrid_global_grab requires GRAB features; remove --only_global")
    if args.pnp_text_only:
        if not args.freeze_host:
            parser.error("--pnp_text_only requires --freeze_host")
        if args.use_host_loss:
            parser.error("--pnp_text_only requires --no_use_host_loss")
        if not args.use_freeze_indices:
            parser.error("--pnp_text_only requires --use_freeze_indices")
        if args.enrichment_space != "global":
            parser.error("--pnp_text_only requires --enrichment_space global")
    if args.num_parts < 1:
        parser.error("--num_parts must be a positive integer")
    if args.residual_gate == "static" and args.enrich_gamma is None:
        parser.error("--residual_gate static requires --enrich_gamma")
    if args.residual_gate == "residual" and args.enrich_gamma is not None:
        parser.error("--enrich_gamma is only valid with --residual_gate static")
    if args.residual_gate_hidden_dim < 1:
        parser.error("--residual_gate_hidden_dim must be a positive integer")
    if args.mixer_dim < 1:
        parser.error("--mixer_dim must be a positive integer")
    if args.mixer_depth < 1:
        parser.error("--mixer_depth must be a positive integer")
    if args.mixer_hidden_part < 1:
        parser.error("--mixer_hidden_part must be a positive integer")
    if args.mixer_hidden_rank < 1:
        parser.error("--mixer_hidden_rank must be a positive integer")
    if args.mixer_hidden_channel < 1:
        parser.error("--mixer_hidden_channel must be a positive integer")
    if args.mixer_hidden_readout < 1:
        parser.error("--mixer_hidden_readout must be a positive integer")
    return args
