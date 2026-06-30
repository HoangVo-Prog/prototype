QCRS_MIXER_VARIANTS = (
    "qcrs_full",
    "mean_pool",
    "flat_mlp",
    "no_film",
    "no_rank_emb",
    "no_slot_emb",
    "no_slotmix",
    "no_rankmix",
    "no_chanmix",
)


_QCRS_MIXER_VARIANT_CONFIGS = {
    "qcrs_full": {
        "mixer_mode": "qcrs",
        "use_film": True,
        "use_rank_emb": True,
        "use_slot_emb": True,
        "use_slotmix": True,
        "use_rankmix": True,
        "use_chanmix": True,
    },
    "mean_pool": {
        "mixer_mode": "mean_pool",
        "use_film": False,
        "use_rank_emb": False,
        "use_slot_emb": False,
        "use_slotmix": False,
        "use_rankmix": False,
        "use_chanmix": False,
    },
    "flat_mlp": {
        "mixer_mode": "flat_mlp",
        "use_film": True,
        "use_rank_emb": True,
        "use_slot_emb": True,
        "use_slotmix": False,
        "use_rankmix": False,
        "use_chanmix": False,
    },
    "no_film": {
        "mixer_mode": "qcrs",
        "use_film": False,
        "use_rank_emb": True,
        "use_slot_emb": True,
        "use_slotmix": True,
        "use_rankmix": True,
        "use_chanmix": True,
    },
    "no_rank_emb": {
        "mixer_mode": "qcrs",
        "use_film": True,
        "use_rank_emb": False,
        "use_slot_emb": True,
        "use_slotmix": True,
        "use_rankmix": True,
        "use_chanmix": True,
    },
    "no_slot_emb": {
        "mixer_mode": "qcrs",
        "use_film": True,
        "use_rank_emb": True,
        "use_slot_emb": False,
        "use_slotmix": True,
        "use_rankmix": True,
        "use_chanmix": True,
    },
    "no_slotmix": {
        "mixer_mode": "qcrs",
        "use_film": True,
        "use_rank_emb": True,
        "use_slot_emb": True,
        "use_slotmix": False,
        "use_rankmix": True,
        "use_chanmix": True,
    },
    "no_rankmix": {
        "mixer_mode": "qcrs",
        "use_film": True,
        "use_rank_emb": True,
        "use_slot_emb": True,
        "use_slotmix": True,
        "use_rankmix": False,
        "use_chanmix": True,
    },
    "no_chanmix": {
        "mixer_mode": "qcrs",
        "use_film": True,
        "use_rank_emb": True,
        "use_slot_emb": True,
        "use_slotmix": True,
        "use_rankmix": True,
        "use_chanmix": False,
    },
}


def qcrs_mixer_variant_config(variant):
    if variant not in _QCRS_MIXER_VARIANT_CONFIGS:
        raise ValueError(
            "--qcrs_mixer_variant must be one of {}; got {!r}".format(
                QCRS_MIXER_VARIANTS,
                variant,
            )
        )
    return dict(_QCRS_MIXER_VARIANT_CONFIGS[variant])


def apply_qcrs_mixer_variant(args):
    variant = getattr(args, "qcrs_mixer_variant", "qcrs_full")
    config = qcrs_mixer_variant_config(variant)
    args.qcrs_mixer_mode = config["mixer_mode"]
    args.qcrs_use_film = config["use_film"]
    args.qcrs_use_rank_emb = config["use_rank_emb"]
    args.qcrs_use_slot_emb = config["use_slot_emb"]
    args.qcrs_use_slotmix = config["use_slotmix"]
    args.qcrs_use_rankmix = config["use_rankmix"]
    args.qcrs_use_chanmix = config["use_chanmix"]
    return config
