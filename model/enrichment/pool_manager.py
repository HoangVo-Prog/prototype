import random
from collections import defaultdict

import torch

from .pool_cache import TargetPoolCacheMixin
from .pool_common import _parse_pool_k_candidates, _pool_transform
from .pool_coverage import TargetPoolCoverageMixin
from .pool_sampling import TargetPoolSamplingMixin
from .pool_selection import TargetPoolSelectionMixin


class TargetPoolManager(
    TargetPoolCacheMixin,
    TargetPoolSelectionMixin,
    TargetPoolCoverageMixin,
    TargetPoolSamplingMixin,
):
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
        self.frozen_query_features = None
        self.frozen_gallery_image_ids = None
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
        use_freeze_indices = getattr(self.args, "use_freeze_indices", False)
        if use_freeze_indices:
            if not self._frozen_index_ready():
                self._build_frozen_index_cache(model)
            if not self.use_shared_k:
                return self._frozen_batch_cache(batch)

        if not self.use_shared_k:
            return self._full_training_set_cache(model, epoch, step)

        if self._should_refresh(epoch, step):
            self._clear_interval_cache()
            self.refresh(model, epoch, step)
            interval_id = self._interval_id(epoch, step)
            interval_cache = self._build_interval_pool_cache(
                model=model,
                batch=batch,
                interval_id=interval_id,
            )
            self.interval_cache = interval_cache
            self.active_interval_id = interval_id
        elif self.interval_cache is not None and "diagnostics" in self.interval_cache:
            self.interval_cache["diagnostics"]["pool_interval_reused"] = 1.0
        if use_freeze_indices:
            return self._frozen_shared_k_batch_cache(batch, self.interval_cache)
        return self.interval_cache

    def _clear_interval_cache(self):
        if self.interval_cache is None:
            return
        self.interval_cache = None
        self.active_interval_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
