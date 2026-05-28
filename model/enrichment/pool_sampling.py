import math

import torch

from .pool_common import _js_distance


class TargetPoolSamplingMixin:
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
