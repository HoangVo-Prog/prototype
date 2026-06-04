import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.reproducibility import seed_worker, seeded_generator

from .pool_common import _PoolImageDataset, _PoolTextDataset, _unwrap_model


class TargetPoolCacheMixin:
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
        self.frozen_query_features = query_features if getattr(self.args, "use_shared_k", False) else None
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

    def _frozen_query_indices(self, batch):
        if "index" not in batch:
            raise ValueError("--use_freeze_indices requires training batches to include dataset indices")

        query_indices = batch["index"].detach().long().cpu()
        if query_indices.numel() == 0:
            raise ValueError("Cannot gather frozen indices for an empty batch")
        if int(query_indices.max().item()) >= self.frozen_rank_indices.shape[0]:
            raise ValueError("Batch query index exceeds the frozen retrieval index size")
        if int(query_indices.min().item()) < 0:
            raise ValueError("Batch query index must be non-negative")
        return query_indices

    def _frozen_batch_cache(self, batch):
        device = self.frozen_cache["host_image_features"].device
        query_indices = self._frozen_query_indices(batch)

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

    def _frozen_shared_k_batch_cache(self, batch, pool_cache):
        query_indices = self._frozen_query_indices(batch)
        top_indices = self._frozen_top_indices_within_pool(query_indices, pool_cache)

        diagnostics = dict(pool_cache.get("diagnostics", {}))
        diagnostics["frozen_indices_used"] = 1.0
        diagnostics["frozen_indices_pool_filtered"] = 1.0
        diagnostics["frozen_index_depth"] = float(self.frozen_index_depth)
        diagnostics["frozen_top_m"] = float(top_indices.shape[1])
        diagnostics["pool_shared_k_used"] = 1.0

        cache = {
            key: value
            for key, value in pool_cache.items()
            if key != "diagnostics"
        }
        cache["top_indices"] = top_indices
        cache["diagnostics"] = diagnostics
        return cache

    def _frozen_top_indices_within_pool(self, query_indices, pool_cache):
        if "host_image_features" not in pool_cache:
            raise ValueError("Shared-K pool cache must include host_image_features")

        device = pool_cache["host_image_features"].device
        pool_size = int(pool_cache["host_image_features"].shape[0])
        top_m = min(int(getattr(self.args, "top_m", 1)), pool_size)
        if top_m < 1:
            raise ValueError("Cannot select frozen top-M from an empty shared K pool")

        query_features = getattr(self, "frozen_query_features", None)
        if query_features is not None:
            # Score only the current shared-K pool; this is equivalent to filtering
            # the frozen full-gallery ranking to pool entries before top-M.
            query_features = query_features.index_select(0, query_indices).to(device)
            query_features = F.normalize(query_features.float(), p=2, dim=-1)
            pool_features = F.normalize(pool_cache["host_image_features"].float(), p=2, dim=-1)
            scores = query_features @ pool_features.t()
            return scores.topk(k=top_m, dim=1, largest=True, sorted=True).indices

        return self._filter_frozen_rank_indices_to_pool(
            query_indices=query_indices,
            pool_cache=pool_cache,
            top_m=top_m,
            device=device,
        )

    def _filter_frozen_rank_indices_to_pool(self, query_indices, pool_cache, top_m, device):
        if "image_ids" not in pool_cache:
            raise ValueError("Shared-K pool cache must include image_ids to filter frozen indices")

        local_index_by_record_index = {}
        for local_index, image_id in enumerate(pool_cache["image_ids"].detach().long().cpu().tolist()):
            record_index = self.record_index_by_image_id.get(int(image_id))
            if self._valid_record_index(record_index):
                local_index_by_record_index[int(record_index)] = int(local_index)

        if not local_index_by_record_index:
            raise ValueError("Shared-K pool image_ids do not match the frozen training index")

        rank_rows = self.frozen_rank_indices.index_select(0, query_indices).cpu()
        filtered_rows = []
        for rank_row in rank_rows.tolist():
            selected = []
            selected_set = set()
            for record_index in rank_row:
                local_index = local_index_by_record_index.get(int(record_index))
                if local_index is None or local_index in selected_set:
                    continue
                selected.append(local_index)
                selected_set.add(local_index)
                if len(selected) == top_m:
                    break
            if len(selected) < top_m:
                raise ValueError(
                    "Frozen index depth does not contain enough shared-K pool entries "
                    "for top-M selection; rebuild with cached frozen query features or "
                    "increase --pool_k."
                )
            filtered_rows.append(selected)

        return torch.tensor(filtered_rows, dtype=torch.long, device=device)

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
