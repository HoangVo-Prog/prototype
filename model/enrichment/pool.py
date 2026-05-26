import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils.reproducibility import seed_worker, seeded_generator


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _pool_transform(img_size):
    import torchvision.transforms as T

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def _js_distance(p, q, eps=1e-12):
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float((0.5 * (kl_pm + kl_qm)).sqrt().item())


def _parse_pool_k_candidates(value):
    if isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = str(value).split(",")

    parsed = []
    for candidate in candidates:
        if str(candidate).strip() == "":
            continue
        parsed.append(int(candidate))
    return sorted({candidate for candidate in parsed if candidate > 0})


class _PoolImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from utils.iotools import read_image

        record = self.records[index]
        image = read_image(record["img_path"])
        if self.transform is not None:
            image = self.transform(image)
        return record["pid"], record["image_id"], image


class _PoolTextDataset(Dataset):
    def __init__(self, records, tokenizer, text_length, truncate):
        self.records = records
        self.tokenizer = tokenizer
        self.text_length = text_length
        self.truncate = truncate

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from datasets.bases import tokenize

        record = self.records[index]
        caption = tokenize(
            record["caption"],
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return record["pid"], record["query_index"], caption


class TargetPoolManager:
    def __init__(self, train_dataset, args, logger=None):
        self.use_shared_k = getattr(args, "use_shared_k", False)
        if self.use_shared_k and args.pool_k < 1:
            raise ValueError("--pool_k must be a positive integer")
        if not (0 < args.positive_ratio_max <= 1):
            raise ValueError("--positive_ratio_max must be in (0, 1]")
        pool_k_mode = getattr(args, "pool_k_mode", "static")
        if self.use_shared_k and pool_k_mode not in ("static", "adaptive"):
            raise ValueError("--pool_k_mode must be either 'static' or 'adaptive'")
        self.args = args
        self.seed = int(getattr(args, "seed", 1))
        self.logger = logger
        self.train_dataset = train_dataset
        self.pool_k_mode = pool_k_mode
        self.pool_k_candidates = (
            _parse_pool_k_candidates(getattr(args, "pool_k_candidates", "512,1024,2048,4096,8192"))
            if self.use_shared_k
            else []
        )
        if self.use_shared_k and not self.pool_k_candidates:
            raise ValueError("--pool_k_candidates must contain at least one positive integer")
        self.records = self._build_unique_records(train_dataset.dataset)
        self.query_records = self._build_query_records(train_dataset.dataset)
        self.records_by_pid = self._build_records_by_pid(self.records)
        self.record_by_image_id = {record["image_id"]: record for record in self.records}
        self.record_index_by_image_id = {
            record["image_id"]: index for index, record in enumerate(self.records)
        }
        self.transform = _pool_transform(args.img_size)
        self.cluster_to_indices = None
        self.cluster_distribution = None
        self.cluster_labels = None
        self.last_refresh_unit = None
        self.active_interval_id = None
        self.interval_cache = None
        self.full_training_cache = None
        self.full_training_interval_id = None
        self.full_training_cache_requests = 0
        self.frozen_cache = None
        self.frozen_rank_indices = None
        self.frozen_index_depth = None
        self.frozen_cache_requests = 0
        self.pool_coverage_counts = [0 for _ in self.records]
        self.rng = random.Random(self.seed)

        if not self.use_shared_k and not getattr(args, "use_freeze_indices", False) and logger is not None:
            logger.info(
                "Target enrichment will select top-M directly from the full training set; "
                "pass --use_shared_k to sample a shared K pool first."
            )
        if getattr(args, "use_freeze_indices", False) and logger is not None:
            if not getattr(args, "freeze_host", False):
                logger.warning(
                    "--use_freeze_indices is intended for a frozen host. If host weights "
                    "change, the precomputed retrieval ranking will become stale."
                )
            if getattr(args, "txt_aug", False):
                logger.warning(
                    "--use_freeze_indices builds rankings from unaugmented captions; "
                    "consider disabling text augmentation for exact frozen-host training."
                )

    def _build_unique_records(self, dataset):
        records = {}
        for pid, image_id, img_path, _ in dataset:
            if image_id not in records:
                records[int(image_id)] = {
                    "pid": int(pid),
                    "image_id": int(image_id),
                    "img_path": img_path,
                }
        return list(records.values())

    def _build_query_records(self, dataset):
        records = []
        for query_index, (pid, _, _, caption) in enumerate(dataset):
            records.append({
                "pid": int(pid),
                "query_index": int(query_index),
                "caption": caption,
            })
        return records

    def _build_records_by_pid(self, records):
        records_by_pid = defaultdict(list)
        for index, record in enumerate(records):
            records_by_pid[int(record["pid"])].append(index)
        return {
            pid: sorted(indices, key=lambda idx: records[idx]["image_id"])
            for pid, indices in records_by_pid.items()
        }

    def get_train_cache(self, model, batch, epoch, step):
        if getattr(self.args, "use_freeze_indices", False):
            if self.frozen_cache is None or self.frozen_rank_indices is None:
                self._build_frozen_index_cache(model)
            return self._frozen_batch_cache(batch)

        if not self.use_shared_k:
            return self._full_training_set_cache(model, epoch, step)

        if self._should_refresh(epoch, step):
            self.refresh(model, epoch, step)
            interval_id = self._interval_id(epoch, step)
            self.interval_cache = self._build_interval_pool_cache(
                model=model,
                batch=batch,
                interval_id=interval_id,
            )
            self.active_interval_id = interval_id
        elif self.interval_cache is not None and "diagnostics" in self.interval_cache:
            self.interval_cache["diagnostics"]["pool_interval_reused"] = 1.0
        return self.interval_cache

    def _build_frozen_index_cache(self, model):
        if not self.records:
            raise ValueError("Cannot build frozen indices from an empty training image pool")
        if not self.query_records:
            raise ValueError("Cannot build frozen indices from an empty training query set")

        cache = self._encode_records(model, self.records, cache_prototypes=True)
        device = cache["host_image_features"].device
        cache["image_ids"] = torch.tensor(
            [record["image_id"] for record in self.records],
            dtype=torch.long,
            device=device,
        )
        cache["pids"] = torch.tensor(
            [record["pid"] for record in self.records],
            dtype=torch.long,
            device=device,
        )

        host_image_features = F.normalize(cache["host_image_features"].float(), p=2, dim=-1)
        query_features = self._encode_text_records(model, self.query_records)
        rank_depth = min(
            max(int(getattr(self.args, "pool_k", 1)), int(getattr(self.args, "top_m", 1))),
            host_image_features.shape[0],
        )
        if rank_depth < 1:
            raise ValueError("Frozen index depth must be positive")

        rank_chunks = []
        batch_size = min(max(1, self.args.test_batch_size), max(1, query_features.shape[0]))
        with torch.no_grad():
            for start in range(0, query_features.shape[0], batch_size):
                query_chunk = query_features[start:start + batch_size].to(device)
                query_chunk = F.normalize(query_chunk.float(), p=2, dim=-1)
                scores = query_chunk @ host_image_features.t()
                rank_chunks.append(
                    scores.topk(k=rank_depth, dim=1, largest=True, sorted=True).indices.cpu()
                )

        self.frozen_cache = cache
        self.frozen_rank_indices = torch.cat(rank_chunks, dim=0)
        self.frozen_index_depth = rank_depth
        self.frozen_cache_requests = 0

        if self.logger is not None:
            self.logger.info(
                "Frozen host retrieval index built: queries={} gallery={} rank_depth={}".format(
                    self.frozen_rank_indices.shape[0],
                    cache["pids"].numel(),
                    rank_depth,
                )
            )

    def _frozen_batch_cache(self, batch):
        if "index" not in batch:
            raise ValueError("--use_freeze_indices requires training batches to include dataset indices")

        device = self.frozen_cache["host_image_features"].device
        query_indices = batch["index"].detach().long().cpu()
        if query_indices.numel() == 0:
            raise ValueError("Cannot gather frozen indices for an empty batch")
        if int(query_indices.max().item()) >= self.frozen_rank_indices.shape[0]:
            raise ValueError("Batch query index exceeds the frozen retrieval index size")
        if int(query_indices.min().item()) < 0:
            raise ValueError("Batch query index must be non-negative")

        top_m = min(int(getattr(self.args, "top_m", 1)), self.frozen_index_depth)
        top_indices = self.frozen_rank_indices.index_select(0, query_indices)[:, :top_m].to(device)

        diagnostics = {
            "pool_interval_id": 0.0,
            "pool_interval_reused": float(self.frozen_cache_requests > 0),
            "pool_k_mode": "frozen_indices",
            "pool_dist_metric": getattr(self.args, "pool_dist_metric", ""),
            "pool_shared_k_used": 0.0,
            "selected_K": float(self.frozen_index_depth),
            "pool_selected_k": float(self.frozen_index_depth),
            "pool_final_pool_size": float(self.frozen_cache["pids"].numel()),
            "pool_num_required_positives": 0.0,
            "pool_num_inserted_positives": 0.0,
            "pool_positive_ratio": 0.0,
            "pool_cluster_distribution_distance": 0.0,
            "pool_missing_positive_count": 0.0,
            "pool_cluster_shortage_count": 0.0,
            "pool_k_valid": 1.0,
            "pool_k_dilute": 1.0,
            "pool_k_dist": 1.0,
            "num_required_positives": 0.0,
            "num_inserted_positives": 0.0,
            "missing_positive_count": 0.0,
            "positive_ratio": 0.0,
            "cluster_distribution_distance": 0.0,
            "final_pool_size": float(self.frozen_cache["pids"].numel()),
            "cluster_shortage_count": 0.0,
            "K_valid": 1.0,
            "K_dilute": 1.0,
            "K_dist": 1.0,
            "frozen_indices_used": 1.0,
            "frozen_index_depth": float(self.frozen_index_depth),
        }

        self.frozen_cache_requests += 1
        cache = {key: value for key, value in self.frozen_cache.items()}
        cache["top_indices"] = top_indices
        cache["diagnostics"] = diagnostics
        return cache

    def _full_training_set_cache(self, model, epoch, step):
        if self._should_refresh_full_training_set_cache(epoch, step):
            self._build_full_training_set_cache(model, epoch, step)

        diagnostics = dict(self.full_training_cache["diagnostics"])
        diagnostics["pool_interval_reused"] = float(self.full_training_cache_requests > 0)
        cache = {
            key: value
            for key, value in self.full_training_cache.items()
            if key != "diagnostics"
        }
        cache["diagnostics"] = diagnostics
        self.full_training_cache_requests += 1
        return cache

    def _should_refresh_full_training_set_cache(self, epoch, step):
        if self.full_training_cache is None:
            return True
        if self.args.recompute_interval == -1:
            return False
        return self._interval_id(epoch, step) != self.full_training_interval_id

    def _build_full_training_set_cache(self, model, epoch, step):
        if not self.records:
            raise ValueError("Cannot build a full-training-set target cache from an empty image pool")

        interval_id = self._interval_id(epoch, step)
        cache = self._encode_records(model, self.records, cache_prototypes=True)
        device = cache["host_image_features"].device
        pool_size = len(self.records)
        cache["image_ids"] = torch.tensor(
            [record["image_id"] for record in self.records],
            dtype=torch.long,
            device=device,
        )
        cache["pids"] = torch.tensor(
            [record["pid"] for record in self.records],
            dtype=torch.long,
            device=device,
        )
        cache["diagnostics"] = {
            "pool_interval_id": float(interval_id),
            "pool_interval_reused": 0.0,
            "pool_k_mode": "full_training_set",
            "pool_dist_metric": getattr(self.args, "pool_dist_metric", ""),
            "selected_K": float(pool_size),
            "pool_selected_k": float(pool_size),
            "pool_final_pool_size": float(pool_size),
            "pool_num_required_positives": 0.0,
            "pool_num_inserted_positives": 0.0,
            "pool_positive_ratio": 0.0,
            "pool_cluster_distribution_distance": 0.0,
            "pool_missing_positive_count": 0.0,
            "pool_cluster_shortage_count": 0.0,
            "pool_k_valid": 1.0,
            "pool_k_dilute": 1.0,
            "pool_k_dist": 1.0,
            "pool_shared_k_used": 0.0,
            "num_required_positives": 0.0,
            "num_inserted_positives": 0.0,
            "missing_positive_count": 0.0,
            "positive_ratio": 0.0,
            "cluster_distribution_distance": 0.0,
            "final_pool_size": float(pool_size),
            "cluster_shortage_count": 0.0,
            "K_valid": 1.0,
            "K_dilute": 1.0,
            "K_dist": 1.0,
        }

        self.full_training_cache = cache
        self.full_training_interval_id = interval_id
        self.full_training_cache_requests = 0
        if self.logger is not None:
            self.logger.info(
                "Target-pool full training cache built: interval={} images={} top_m={}".format(
                    interval_id,
                    pool_size,
                    getattr(self.args, "top_m", 1),
                )
            )

    def _interval_unit(self, epoch, step):
        return epoch if self.args.recompute_level == "epoch" else step

    def _interval_id(self, epoch, step):
        interval = self.args.recompute_interval
        if interval == -1:
            return 0
        if interval < 1:
            raise ValueError("--recompute_interval must be -1 or a positive integer")
        unit = self._interval_unit(epoch, step)
        return (unit - 1) // interval

    def _should_refresh(self, epoch, step):
        if self.cluster_to_indices is None or self.interval_cache is None:
            return True
        if self.args.recompute_interval == -1:
            return False
        return self._interval_id(epoch, step) != self.active_interval_id

    def refresh(self, model, epoch, step):
        self._estimate_training_distribution(model)
        self.last_refresh_unit = self._interval_unit(epoch, step)
        if self.logger is not None:
            self.logger.info(
                "Target-pool training distribution refreshed: C={} level={} unit={}".format(
                    len(self.cluster_to_indices), self.args.recompute_level, self.last_refresh_unit
                )
            )

    def _estimate_training_distribution(self, model):
        features = self._encode_records(model, self.records, cache_prototypes=False)["host_image_features"].cpu()
        num_images = features.shape[0]
        num_clusters = min(max(1, self.args.pool_clusters), num_images)

        if num_clusters == 1:
            labels = torch.zeros(num_images, dtype=torch.long)
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_clusters, random_state=self.seed % 2**32, n_init=10)
            labels = torch.tensor(kmeans.fit_predict(features.numpy()), dtype=torch.long)

        cluster_to_indices = defaultdict(list)
        for index, label in enumerate(labels.tolist()):
            cluster_to_indices[int(label)].append(index)

        counts = torch.tensor([len(cluster_to_indices[c]) for c in range(num_clusters)], dtype=torch.float)
        self.cluster_distribution = counts / counts.sum()
        self.cluster_to_indices = dict(cluster_to_indices)
        self.cluster_labels = labels

    def _build_batch_pool_cache(self, model, batch):
        return self._build_interval_pool_cache(model, batch, interval_id=0)

    def _build_interval_pool_cache(self, model, batch, interval_id):
        positives = self._required_identity_anchors()
        pool_k, distractor_indices, diagnostics = self._select_pool_k_and_distractors(
            positives=positives,
            interval_id=interval_id,
        )
        diagnostics["pool_interval_id"] = float(interval_id)
        diagnostics["pool_interval_reused"] = 0.0
        diagnostics["pool_k_mode"] = getattr(self.args, "pool_k_mode", "static")
        diagnostics["pool_dist_metric"] = self.args.pool_dist_metric
        diagnostics["pool_shared_k_used"] = 1.0
        diagnostics["selected_K"] = float(pool_k)
        diagnostics["pool_selected_k"] = float(pool_k)
        diagnostics["num_required_positives"] = diagnostics["pool_num_required_positives"]
        diagnostics["num_inserted_positives"] = diagnostics["pool_num_inserted_positives"]
        diagnostics["missing_positive_count"] = diagnostics["pool_missing_positive_count"]
        diagnostics["positive_ratio"] = diagnostics["pool_positive_ratio"]
        diagnostics["cluster_distribution_distance"] = diagnostics["pool_cluster_distribution_distance"]
        diagnostics["final_pool_size"] = diagnostics["pool_final_pool_size"]
        diagnostics["cluster_shortage_count"] = diagnostics["pool_cluster_shortage_count"]
        diagnostics["K_valid"] = diagnostics["pool_k_valid"]
        diagnostics["K_dilute"] = diagnostics["pool_k_dilute"]
        diagnostics["K_dist"] = diagnostics["pool_k_dist"]
        diagnostics["K_cover"] = diagnostics["pool_k_cover"]

        if diagnostics["pool_final_pool_size"] != float(pool_k):
            raise ValueError(
                "Target pool construction failed: selected_K={} but only {} unique images "
                "could be inserted. Reduce --pool_k/--pool_k_candidates or add more "
                "training images.".format(pool_k, int(diagnostics["pool_final_pool_size"]))
            )

        positive_record_items = [
            item for item in positives if self._valid_record_index(item["record_index"])
        ]
        fallback_positive_items = [
            item for item in positives if not self._valid_record_index(item["record_index"])
        ]
        distractor_records = [self.records[index] for index in distractor_indices]
        cache = self._build_cache_from_interval_items(
            model=model,
            batch=batch,
            positive_record_items=positive_record_items,
            fallback_positive_items=fallback_positive_items,
            distractor_records=distractor_records,
        )
        covered_record_indices = [
            item["record_index"]
            for item in positive_record_items
            if self._valid_record_index(item["record_index"])
        ] + list(distractor_indices)
        self._mark_pool_coverage(covered_record_indices)
        diagnostics["pool_coverage_seen"] = float(self._coverage_seen_count())
        diagnostics["pool_coverage_remaining"] = float(
            max(0, len(self.records) - self._coverage_seen_count())
        )
        diagnostics["coverage_seen"] = diagnostics["pool_coverage_seen"]
        diagnostics["coverage_remaining"] = diagnostics["pool_coverage_remaining"]
        cache["diagnostics"] = diagnostics
        self._warn_if_constraints_fail(diagnostics)
        self._log_interval_cache(diagnostics)
        return cache

    def _valid_record_index(self, record_index):
        return record_index is not None and 0 <= record_index < len(self.records)

    def _available_pool_capacity(self, positives):
        fallback_positives = [
            item for item in positives if not self._valid_record_index(item["record_index"])
        ]
        return len(self.records) + len(fallback_positives)

    def _select_pool_k_and_distractors(self, positives, interval_id):
        pool_k_mode = getattr(self.args, "pool_k_mode", "static")
        if pool_k_mode == "adaptive":
            return self._select_adaptive_pool_k(positives, interval_id)

        pool_k = self.args.pool_k
        self._check_static_pool_k(pool_k, positives)
        distractor_indices, diagnostics = self._sample_distractor_indices(positives, pool_k)
        return pool_k, distractor_indices, diagnostics

    def _check_static_pool_k(self, pool_k, positives):
        if len(positives) > pool_k:
            raise ValueError(
                "K_valid failed: pool_k={} but shared identity coverage requires {} "
                "unique identity anchors. Increase --pool_k or use --pool_k_mode "
                "adaptive.".format(pool_k, len(positives))
            )
        capacity = self._available_pool_capacity(positives)
        if pool_k > capacity:
            raise ValueError(
                "Target pool construction impossible: pool_k={} exceeds the available "
                "unique training image capacity {} for this interval.".format(pool_k, capacity)
            )

    def _candidate_pool_ks(self, positives):
        candidates = list(getattr(self, "pool_k_candidates", None) or _parse_pool_k_candidates(
            getattr(self.args, "pool_k_candidates", "512,1024,2048,4096,8192")
        ))
        capacity = self._available_pool_capacity(positives)
        if capacity > 0 and not any(candidate <= capacity for candidate in candidates):
            candidates.append(capacity)
        return sorted({candidate for candidate in candidates if candidate > 0})

    def _simulate_candidate_pool(self, positives, pool_k, rng_state):
        self.rng.setstate(rng_state)
        distractor_indices, diagnostics = self._sample_distractor_indices(positives, pool_k)
        next_state = self.rng.getstate()
        return distractor_indices, diagnostics, next_state

    def _select_adaptive_pool_k(self, positives, interval_id):
        capacity = self._available_pool_capacity(positives)
        if len(positives) > capacity:
            raise ValueError(
                "K_valid failed: active interval requires {} positives but only {} "
                "unique images are available.".format(len(positives), capacity)
            )

        k_valid = max(1, len(positives))
        k_dilute = max(k_valid, int(math.ceil(len(positives) / self.args.positive_ratio_max)))
        k_cover = max(k_valid, self._coverage_k_target())
        candidates = self._candidate_pool_ks(positives)
        candidates.extend([k_valid, k_dilute, k_cover])
        candidates = sorted({
            candidate for candidate in candidates
            if candidate > 0 and candidate <= max(capacity, 1)
        })
        if not candidates:
            raise ValueError("--pool_k_candidates must contain at least one positive integer")

        initial_state = self.rng.getstate()
        k_dist = 0
        k_dist_diagnostics = None
        for pool_k in candidates:
            if pool_k > capacity:
                continue
            distractors, diagnostics, next_state = self._simulate_candidate_pool(
                positives, pool_k, initial_state
            )
            if diagnostics["pool_final_pool_size"] != float(pool_k):
                continue
            if not diagnostics["pool_k_valid"]:
                continue
            if diagnostics["pool_k_dist"]:
                k_dist = pool_k
                k_dist_diagnostics = diagnostics
                break

        selected_k = max(k_valid, k_dilute, k_dist, k_cover)
        capacity_limited = False
        if selected_k > capacity:
            if k_valid > capacity:
                raise ValueError(
                    "K_valid failed: no adaptive K can include {} identity anchors "
                    "within capacity {}. Reduce the interval size or add training images.".format(
                        len(positives), capacity
                    )
                )
            selected_k = capacity
            capacity_limited = True

        distractors, diagnostics, next_state = self._simulate_candidate_pool(
            positives, selected_k, initial_state
        )
        if diagnostics["pool_final_pool_size"] != float(selected_k):
            raise ValueError(
                "Adaptive target pool construction failed: selected_K={} but only {} "
                "unique images could be inserted.".format(
                    selected_k,
                    int(diagnostics["pool_final_pool_size"]),
                )
            )

        self.rng.setstate(next_state)
        diagnostics["pool_adaptive_fallback"] = float(k_dist == 0 or capacity_limited)
        diagnostics["pool_k_valid_target"] = float(k_valid)
        diagnostics["pool_k_dilute_target"] = float(k_dilute)
        diagnostics["pool_k_dist_target"] = float(k_dist)
        diagnostics["pool_k_cover_target"] = float(k_cover)
        diagnostics["pool_k_dist_found"] = float(k_dist > 0)
        diagnostics["pool_k_capacity_limited"] = float(capacity_limited)

        if self.logger is not None:
            self.logger.info(
                "Adaptive target pool interval {} selected K=max({}, {}, {}, {})={}".format(
                    interval_id,
                    k_valid,
                    k_dilute,
                    k_dist,
                    k_cover,
                    selected_k,
                )
            )
            if k_dist == 0:
                self.logger.warning(
                    "Adaptive target pool interval {} could not find K_dist within "
                    "candidates {}; selected K=max(K_valid,K_dilute,K_cover).".format(
                        interval_id,
                        candidates,
                    )
                )
            elif (
                k_dist_diagnostics is not None
                and selected_k != k_dist
                and diagnostics["pool_cluster_distribution_distance"] > self.args.pool_dist_threshold
            ):
                self.logger.warning(
                    "Adaptive target pool interval {} found K_dist={} but selected_K={} "
                    "from max(...) has distance={:.4f}; consider adding selected_K to "
                    "--pool_k_candidates for a direct distribution sweep.".format(
                        interval_id,
                        k_dist,
                        selected_k,
                        diagnostics["pool_cluster_distribution_distance"],
                    )
                )
            if capacity_limited:
                self.logger.warning(
                    "Adaptive target pool interval {} was capacity-limited at K={}.".format(
                        interval_id,
                        selected_k,
                    )
                )
        return selected_k, distractors, diagnostics

    def _required_positives(self, batch):
        image_ids = batch["image_ids"].detach().long().tolist()
        pids = batch["pids"].detach().long().tolist()
        positives = []
        seen = set()
        for batch_position, image_id in enumerate(image_ids):
            if image_id in seen:
                continue
            seen.add(image_id)
            record_index = self.record_index_by_image_id.get(image_id)
            cluster_id = None
            if self._valid_record_index(record_index) and self.cluster_labels is not None:
                cluster_id = int(self.cluster_labels[record_index].item())
            positives.append({
                "batch_position": batch_position,
                "image_id": int(image_id),
                "pid": int(pids[batch_position]),
                "record_index": record_index,
                "cluster_id": cluster_id,
            })
        return positives

    def _coverage_horizon(self):
        return max(1, int(getattr(self.args, "pool_coverage_epochs", 15)))

    def _coverage_seen_count(self):
        counts = getattr(self, "pool_coverage_counts", [])
        return sum(1 for count in counts if count > 0)

    def _coverage_k_target(self):
        if not self.records:
            return 0
        records_by_pid = getattr(self, "records_by_pid", None)
        if records_by_pid is None:
            records_by_pid = self._build_records_by_pid(self.records)
            self.records_by_pid = records_by_pid

        horizon = self._coverage_horizon()
        identity_slots = len(records_by_pid)
        covered_by_identity_rotation = sum(
            min(len(indices), horizon) for indices in records_by_pid.values()
        )
        remaining_after_identity_rotation = max(
            0, len(self.records) - covered_by_identity_rotation
        )
        extra_slots = int(math.ceil(remaining_after_identity_rotation / horizon))
        return min(len(self.records), identity_slots + extra_slots)

    def _coverage_preferred_indices(self, candidates, count):
        if count <= 0:
            return []
        coverage_counts = getattr(self, "pool_coverage_counts", None)
        if coverage_counts is None:
            coverage_counts = [0 for _ in self.records]
            self.pool_coverage_counts = coverage_counts
        if not hasattr(self, "rng"):
            self.rng = random.Random(getattr(self, "seed", int(getattr(self.args, "seed", 1))))
        decorated = [
            (coverage_counts[index], self.rng.random(), index)
            for index in candidates
        ]
        decorated.sort()
        return [index for _, _, index in decorated[:count]]

    def _coverage_preferred_index(self, candidates):
        picks = self._coverage_preferred_indices(candidates, 1)
        return picks[0] if picks else None

    def _mark_pool_coverage(self, record_indices):
        if not hasattr(self, "pool_coverage_counts"):
            self.pool_coverage_counts = [0 for _ in self.records]
        for index in record_indices:
            if 0 <= index < len(self.pool_coverage_counts):
                self.pool_coverage_counts[index] += 1

    def _required_identity_anchors(self):
        records_by_pid = getattr(self, "records_by_pid", None)
        if records_by_pid is None:
            records_by_pid = self._build_records_by_pid(self.records)
            self.records_by_pid = records_by_pid

        anchors = []
        for pid in sorted(records_by_pid):
            record_index = self._coverage_preferred_index(records_by_pid[pid])
            if record_index is None:
                continue
            record = self.records[record_index]
            cluster_id = None
            if self.cluster_labels is not None:
                cluster_id = int(self.cluster_labels[record_index].item())
            anchors.append({
                "batch_position": None,
                "image_id": int(record["image_id"]),
                "pid": int(record["pid"]),
                "record_index": record_index,
                "cluster_id": cluster_id,
            })
        return anchors

    def _target_cluster_quotas(self, pool_k):
        raw = self.cluster_distribution * pool_k
        quotas = torch.floor(raw).long()
        while quotas.sum().item() < pool_k:
            deficit = raw - quotas.float()
            quotas[int(deficit.argmax().item())] += 1
        while quotas.sum().item() > pool_k:
            surplus = quotas.float() - raw
            idx = int(surplus.argmax().item())
            if quotas[idx] == 0:
                break
            quotas[idx] -= 1
        return quotas

    def _sample_distractor_indices(self, positives, pool_k):
        num_clusters = len(self.cluster_to_indices)
        quotas = self._target_cluster_quotas(pool_k)
        positive_record_indices = {
            item["record_index"] for item in positives if self._valid_record_index(item["record_index"])
        }
        positive_cluster_counts = torch.zeros(num_clusters, dtype=torch.long)
        missing_positive_count = 0
        for item in positives:
            if item["cluster_id"] is None:
                missing_positive_count += 1
            else:
                positive_cluster_counts[item["cluster_id"]] += 1

        remaining_quotas = torch.clamp(quotas - positive_cluster_counts, min=0)
        selected = []
        selected_set = set(positive_record_indices)
        cluster_shortage_count = 0
        distractor_capacity = max(0, pool_k - len(positives))
        for cluster_id in range(num_clusters):
            if len(selected) >= distractor_capacity:
                break
            candidates = [
                idx for idx in self.cluster_to_indices[cluster_id]
                if idx not in selected_set
            ]
            requested = int(remaining_quotas[cluster_id].item())
            quota = min(requested, len(candidates), distractor_capacity - len(selected))
            cluster_shortage_count += max(0, requested - quota)
            if quota > 0:
                picks = self._coverage_preferred_indices(candidates, quota)
                selected.extend(picks)
                selected_set.update(picks)

        while len(positives) + len(selected) < pool_k:
            counts = self._final_cluster_counts(positives, selected, num_clusters)
            empirical = counts.float() / max(pool_k, 1)
            deficits = self.cluster_distribution - empirical
            ordered_clusters = torch.argsort(deficits, descending=True).tolist()
            added = False
            for cluster_id in ordered_clusters:
                candidates = [
                    idx for idx in self.cluster_to_indices[cluster_id]
                    if idx not in selected_set
                ]
                if not candidates:
                    continue
                pick = self._coverage_preferred_index(candidates)
                selected.append(pick)
                selected_set.add(pick)
                added = True
                break
            if not added:
                break

        final_counts = self._final_cluster_counts(positives, selected, num_clusters)
        final_pool_size = len(positives) + len(selected)
        positive_ratio = len(positives) / max(pool_k, 1)
        distance = self._distribution_distance(final_counts, final_pool_size)
        k_cover = self._coverage_k_target()
        coverage_seen_before = self._coverage_seen_count()
        diagnostics = {
            "pool_selected_k": float(pool_k),
            "pool_num_required_positives": float(len(positives)),
            "pool_num_inserted_positives": float(len(positives)),
            "pool_positive_ratio": float(positive_ratio),
            "pool_cluster_distribution_distance": float(distance),
            "pool_final_pool_size": float(final_pool_size),
            "pool_missing_positive_count": float(missing_positive_count),
            "pool_cluster_shortage_count": float(cluster_shortage_count),
            "pool_k_valid": float(len(positives) <= pool_k),
            "pool_k_dilute": float(positive_ratio <= self.args.positive_ratio_max),
            "pool_k_dist": float(distance <= self.args.pool_dist_threshold),
            "pool_k_cover": float(pool_k >= k_cover),
            "pool_k_cover_target": float(k_cover),
            "pool_coverage_horizon": float(self._coverage_horizon()),
            "pool_coverage_seen_before": float(coverage_seen_before),
            "pool_coverage_seen": float(coverage_seen_before),
            "pool_coverage_remaining": float(max(0, len(self.records) - coverage_seen_before)),
        }
        return selected, diagnostics

    def _final_cluster_counts(self, positives, distractor_indices, num_clusters):
        counts = torch.zeros(num_clusters, dtype=torch.long)
        for item in positives:
            if item["cluster_id"] is not None:
                counts[item["cluster_id"]] += 1
        for index in distractor_indices:
            counts[int(self.cluster_labels[index].item())] += 1
        return counts

    def _distribution_distance(self, final_counts, final_pool_size):
        if final_pool_size == 0:
            return math.inf
        empirical = final_counts.float() / final_pool_size
        if self.args.pool_dist_metric == "js":
            return _js_distance(empirical, self.cluster_distribution)
        return float(torch.abs(empirical - self.cluster_distribution).sum().item())

    def _warn_if_constraints_fail(self, diagnostics):
        if self.logger is None:
            return
        if diagnostics["pool_k_valid"] < 1:
            self.logger.warning(
                "K_valid warning: selected_K={} is smaller than required positives={}".format(
                    int(diagnostics["pool_selected_k"]),
                    int(diagnostics["pool_num_required_positives"]),
                )
            )
        if diagnostics["pool_positive_ratio"] > self.args.positive_ratio_max:
            self.logger.warning(
                "K_dilute warning: positive_ratio={:.4f} exceeds eta={:.4f}".format(
                    diagnostics["pool_positive_ratio"], self.args.positive_ratio_max
                )
            )
        if diagnostics["pool_cluster_distribution_distance"] > self.args.pool_dist_threshold:
            self.logger.warning(
                "K_dist warning: {} distance={:.4f} exceeds epsilon={:.4f}".format(
                    self.args.pool_dist_metric,
                    diagnostics["pool_cluster_distribution_distance"],
                    self.args.pool_dist_threshold,
                )
            )
        if diagnostics.get("pool_k_cover", 1.0) < 1:
            self.logger.warning(
                "K_cover warning: selected_K={} is smaller than the {}-pool coverage "
                "target {}".format(
                    int(diagnostics["pool_selected_k"]),
                    int(diagnostics.get("pool_coverage_horizon", 0)),
                    int(diagnostics.get("pool_k_cover_target", 0)),
                )
            )
        if diagnostics["pool_missing_positive_count"] > 0:
            self.logger.warning(
                "Target-pool warning: {} required positives were missing cluster assignments".format(
                    int(diagnostics["pool_missing_positive_count"])
                )
            )
        if diagnostics.get("pool_cluster_shortage_count", 0.0) > 0:
            self.logger.warning(
                "Target-pool quota warning: {} requested cluster slots were unavailable "
                "and were filled from other clusters when possible".format(
                    int(diagnostics["pool_cluster_shortage_count"])
                )
            )

    def _log_interval_cache(self, diagnostics):
        if self.logger is None:
            return
        self.logger.info(
            "Target-pool interval cache built: interval={} mode={} selected_K={} "
            "identity_anchors={} ratio={:.4f} dist={:.4f} final_size={} "
            "coverage={}/{}".format(
                int(diagnostics["pool_interval_id"]),
                diagnostics["pool_k_mode"],
                int(diagnostics["pool_selected_k"]),
                int(diagnostics["pool_num_inserted_positives"]),
                diagnostics["pool_positive_ratio"],
                diagnostics["pool_cluster_distribution_distance"],
                int(diagnostics["pool_final_pool_size"]),
                int(diagnostics.get("pool_coverage_seen", 0)),
                len(self.records),
            )
        )

    def _encode_batch_images(self, model, images, positions):
        if not positions:
            return None
        position_tensor = torch.tensor(positions, dtype=torch.long, device=images.device)
        core_model = _unwrap_model(model)
        was_training = core_model.training
        core_model.eval()
        with torch.no_grad():
            cache = core_model.encode_target_image_cache(images.index_select(0, position_tensor))
        if was_training:
            core_model.train()
        return cache

    def _build_cache_from_interval_items(
        self,
        model,
        batch,
        positive_record_items,
        fallback_positive_items,
        distractor_records,
    ):
        cache_chunks = []
        image_ids = []
        pids = []
        cluster_ids = []

        positive_records = [self.records[item["record_index"]] for item in positive_record_items]
        if positive_records:
            cache_chunks.append(self._encode_records(model, positive_records))
            image_ids.extend([item["image_id"] for item in positive_record_items])
            pids.extend([item["pid"] for item in positive_record_items])
            cluster_ids.extend([
                item["cluster_id"] if item["cluster_id"] is not None else -1
                for item in positive_record_items
            ])

        if fallback_positive_items:
            positions = [item["batch_position"] for item in fallback_positive_items]
            cache_chunks.append(self._encode_batch_images(model, batch["images"], positions))
            image_ids.extend([item["image_id"] for item in fallback_positive_items])
            pids.extend([item["pid"] for item in fallback_positive_items])
            cluster_ids.extend([
                item["cluster_id"] if item["cluster_id"] is not None else -1
                for item in fallback_positive_items
            ])

        if distractor_records:
            cache_chunks.append(self._encode_records(model, distractor_records))
            image_ids.extend([record["image_id"] for record in distractor_records])
            pids.extend([record["pid"] for record in distractor_records])
            cluster_ids.extend([
                int(self.cluster_labels[self.record_index_by_image_id[record["image_id"]]].item())
                for record in distractor_records
            ])

        cache_chunks = [chunk for chunk in cache_chunks if chunk is not None]
        if not cache_chunks:
            raise ValueError("Target pool construction produced an empty pool")

        cache = {}
        for key in cache_chunks[0].keys():
            cache[key] = torch.cat([chunk[key] for chunk in cache_chunks], dim=0)
        device = cache["host_image_features"].device
        cache["image_ids"] = torch.tensor(image_ids, dtype=torch.long, device=device)
        cache["pids"] = torch.tensor(pids, dtype=torch.long, device=device)
        cache["cluster_ids"] = torch.tensor(cluster_ids, dtype=torch.long, device=device)
        return self._shuffle_cache(cache)

    def _merge_positive_and_distractor_cache(
        self,
        positive_cache,
        positive_ids,
        positive_pids,
        distractor_cache,
        distractor_records,
    ):
        if positive_cache is None and distractor_cache is None:
            raise ValueError("Target pool construction produced an empty pool")
        if positive_cache is None:
            cache = {key: value for key, value in distractor_cache.items()}
            device = cache["host_image_features"].device
        elif distractor_cache is None:
            cache = {key: value for key, value in positive_cache.items()}
            device = cache["host_image_features"].device
        else:
            cache = {
                key: torch.cat([positive_cache[key], distractor_cache[key]], dim=0)
                for key in positive_cache.keys()
            }
            device = cache["host_image_features"].device

        image_ids = list(positive_ids) + [record["image_id"] for record in distractor_records]
        pids = list(positive_pids) + [record["pid"] for record in distractor_records]
        cache["image_ids"] = torch.tensor(image_ids, dtype=torch.long, device=device)
        cache["pids"] = torch.tensor(pids, dtype=torch.long, device=device)
        return self._shuffle_cache(cache)

    def _shuffle_cache(self, cache):
        pool_size = cache["pids"].numel()
        order = torch.tensor(
            self.rng.sample(range(pool_size), pool_size),
            dtype=torch.long,
            device=cache["pids"].device,
        )
        return {
            key: value.index_select(0, order) if torch.is_tensor(value) else value
            for key, value in cache.items()
        }

    def _encode_records(self, model, records, cache_prototypes=True):
        core_model = _unwrap_model(model)
        device = next(core_model.parameters()).device
        was_training = core_model.training
        core_model.eval()
        dataset = _PoolImageDataset(records, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=min(max(1, self.args.test_batch_size), max(1, len(records))),
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            generator=seeded_generator(self.seed + 801),
        )

        chunks = []
        with torch.no_grad():
            for _, _, images in loader:
                images = images.to(device)
                chunks.append(core_model.encode_target_image_cache(images, cache_prototypes=cache_prototypes))

        if was_training:
            core_model.train()

        merged = {}
        for key in chunks[0].keys():
            merged[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
        return merged

    def _encode_text_records(self, model, records):
        core_model = _unwrap_model(model)
        device = next(core_model.parameters()).device
        was_training = core_model.training
        core_model.eval()

        tokenizer = getattr(self.train_dataset, "tokenizer", None)
        if tokenizer is None:
            from utils.simple_tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer()
        dataset = _PoolTextDataset(
            records,
            tokenizer=tokenizer,
            text_length=getattr(self.train_dataset, "text_length", self.args.text_length),
            truncate=getattr(self.train_dataset, "truncate", True),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(max(1, self.args.test_batch_size), max(1, len(records))),
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            generator=seeded_generator(self.seed + 901),
        )

        chunks = []
        with torch.no_grad():
            for _, _, captions in loader:
                captions = captions.to(device)
                chunks.append(core_model.encode_text(captions).detach().cpu())

        if was_training:
            core_model.train()

        return torch.cat(chunks, dim=0)
