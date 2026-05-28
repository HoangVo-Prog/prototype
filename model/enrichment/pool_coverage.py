import math
import random


class TargetPoolCoverageMixin:
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
