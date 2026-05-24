import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


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
    parser.add_argument("--log_period", default=20)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--pretrain", type=str, default="")


    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')
    parser.add_argument("--txt_aug", default=True, action='store_true')
    
    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='tal+cid', help="which loss to use ['cid, tal']")
    parser.add_argument("--use_host_loss", type=str2bool, nargs="?", const=True, default=True,
                        help="enable host cid/tal loss in the optimized objective")
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
    parser.add_argument("--lr_total_epoch", type=int, default=-1,
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
    parser.add_argument("--enrichment_space", type=str, default="global",
                        choices=["global", "grab"],
                        help="feature space to enrich when target_enrichment is enabled")
    parser.add_argument("--pool_k_mode", type=str, default="static",
                        choices=["static", "adaptive"],
                        help="static uses --pool_k; adaptive computes K=max(K_valid,K_dilute,K_dist)")
    parser.add_argument("--pool_k", type=int, default=1024,
                        help="number of images in the training pseudo-target pool")
    parser.add_argument("--pool_k_candidates", type=str, default="512,1024,2048,4096,8192",
                        help="comma-separated candidate K values used when --pool_k_mode adaptive")
    parser.add_argument("--top_m", type=int, default=32,
                        help="number of host-ranked local images used for enrichment")
    parser.add_argument("--robust_hard_k", "--hard_neg_k", dest="robust_hard_k",
                        type=int, default=32,
                        help="number of raw-score hard negatives R used by robust margin loss")
    parser.add_argument("--num_parts", type=int, default=6,
                        help="number of horizontal part prototypes per image")
    parser.add_argument("--pool_clusters", type=int, default=16,
                        help="visual clusters used for distribution-preserving pool sampling")
    parser.add_argument("--positive_ratio_max", "--eta", dest="positive_ratio_max",
                        type=float, default=0.5,
                        help="maximum allowed required-positive ratio in a target pool")
    parser.add_argument("--pool_dist_metric", type=str, default="l1",
                        choices=["l1", "js"],
                        help="distance metric for final-pool vs train-set cluster distribution")
    parser.add_argument("--pool_dist_threshold", "--epsilon", dest="pool_dist_threshold",
                        type=float, default=0.25,
                        help="warning threshold for target-pool cluster distribution distance")
    parser.add_argument("--enrich_gamma", type=float, default=0.1,
                        help="residual strength for enriched text features")
    parser.add_argument("--recompute_level", type=str, default="epoch",
                        choices=["epoch", "step"],
                        help="unit used by recompute_interval for target-pool refresh")
    parser.add_argument("--recompute_interval", type=int, default=1,
                        help="-1 computes the target pool once; otherwise refresh every N epochs or steps")
    parser.add_argument("--pool_interval", dest="recompute_interval", type=int,
                        default=argparse.SUPPRESS,
                        help="alias for --recompute_interval")
    
    ######################## target-aware loss settings ########################
    parser.add_argument("--lambda_att", type=float, default=0.1,
                        help="weight for query-aware prototype evidence loss")
    parser.add_argument("--lambda_ret", type=float, default=1.0,
                        help="weight for the target-pool retrieval loss")
    parser.add_argument("--lambda_rob", type=float, default=0.1,
                        help="weight for robust no-harm/margin-gain loss")
    parser.add_argument("--lambda_gain", type=float, default=1.0,
                        help="weight for the margin-gain term inside the robust loss")
    parser.add_argument("--att_margin", type=float, default=0.1,
                        help="margin for query-aware prototype evidence loss")
    parser.add_argument("--gain_margin", type=float, default=0.01,
                        help="required enriched-vs-raw retrieval margin gain")
    parser.add_argument("--use_target_retrieval_loss", type=str2bool, nargs="?", const=True, default=False,
                        help="enable the primary target-pool retrieval loss")
    parser.add_argument("--use_target_attention_loss", type=str2bool, nargs="?", const=True, default=False,
                        help="enable the query-aware prototype evidence loss")
    parser.add_argument("--use_target_robust_loss", type=str2bool, nargs="?", const=True, default=False,
                        help="enable the robust no-harm and margin-gain loss")
    
    args = parser.parse_args()
    return args
