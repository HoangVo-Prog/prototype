import math

from .pool_common import _parse_pool_k_candidates


class TargetPoolSelectionMixin:
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
