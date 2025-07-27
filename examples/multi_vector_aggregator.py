#!/usr/bin/env python3
"""
Multi-Vector Aggregator for Fat Embeddings
==========================================

This module implements aggregation strategies for multi-vector embeddings,
similar to ColPali's approach where multiple patch vectors represent a single document.

Key features:
- MaxSim aggregation (take maximum similarity across patches)
- Voting-based aggregation (count patch matches)
- Weighted aggregation (attention-score weighted)
- Spatial clustering of matching patches
- Document-level result consolidation
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PatchResult:
    """Represents a single patch search result."""

    patch_id: int
    image_name: str
    image_path: str
    coordinates: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    attention_score: float
    scale: float
    metadata: dict[str, Any]


@dataclass
class AggregatedResult:
    """Represents an aggregated document-level result."""

    image_name: str
    image_path: str
    doc_score: float
    patch_count: int
    best_patch: PatchResult
    all_patches: list[PatchResult]
    aggregation_method: str
    spatial_clusters: list[list[PatchResult]] | None = None


class MultiVectorAggregator:
    """
    Aggregates multiple patch-level results into document-level results.
    """

    def __init__(
        self,
        aggregation_method: str = "maxsim",
        spatial_clustering: bool = True,
        cluster_distance_threshold: float = 100.0,
    ):
        """
        Initialize the aggregator.

        Args:
            aggregation_method: "maxsim", "voting", "weighted", or "mean"
            spatial_clustering: Whether to cluster spatially close patches
            cluster_distance_threshold: Distance threshold for spatial clustering
        """
        self.aggregation_method = aggregation_method
        self.spatial_clustering = spatial_clustering
        self.cluster_distance_threshold = cluster_distance_threshold

    def aggregate_results(
        self, search_results: list[dict[str, Any]], top_k: int = 10
    ) -> list[AggregatedResult]:
        """
        Aggregate patch-level search results into document-level results.

        Args:
            search_results: List of search results from LeannSearcher
            top_k: Number of top documents to return

        Returns:
            List of aggregated document results
        """
        # Group results by image
        image_groups = defaultdict(list)

        for result in search_results:
            metadata = result.metadata
            if "image_name" in metadata and "patch_id" in metadata:
                patch_result = PatchResult(
                    patch_id=metadata["patch_id"],
                    image_name=metadata["image_name"],
                    image_path=metadata["image_path"],
                    coordinates=tuple(metadata["coordinates"]),
                    score=result.score,
                    attention_score=metadata.get("attention_score", 0.0),
                    scale=metadata.get("scale", 1.0),
                    metadata=metadata,
                )
                image_groups[metadata["image_name"]].append(patch_result)

        # Aggregate each image group
        aggregated_results = []
        for image_name, patches in image_groups.items():
            if len(patches) == 0:
                continue

            agg_result = self._aggregate_image_patches(image_name, patches)
            aggregated_results.append(agg_result)

        # Sort by aggregated score and return top-k
        aggregated_results.sort(key=lambda x: x.doc_score, reverse=True)
        return aggregated_results[:top_k]

    def _aggregate_image_patches(
        self, image_name: str, patches: list[PatchResult]
    ) -> AggregatedResult:
        """Aggregate patches for a single image."""

        if self.aggregation_method == "maxsim":
            doc_score = max(patch.score for patch in patches)
            best_patch = max(patches, key=lambda p: p.score)

        elif self.aggregation_method == "voting":
            # Count patches above threshold
            threshold = np.percentile([p.score for p in patches], 75)
            doc_score = sum(1 for patch in patches if patch.score >= threshold)
            best_patch = max(patches, key=lambda p: p.score)

        elif self.aggregation_method == "weighted":
            # Weight by attention scores
            total_weighted_score = sum(p.score * p.attention_score for p in patches)
            total_weights = sum(p.attention_score for p in patches)
            doc_score = total_weighted_score / max(total_weights, 1e-8)
            best_patch = max(patches, key=lambda p: p.score * p.attention_score)

        elif self.aggregation_method == "mean":
            doc_score = np.mean([patch.score for patch in patches])
            best_patch = max(patches, key=lambda p: p.score)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Spatial clustering if enabled
        spatial_clusters = None
        if self.spatial_clustering:
            spatial_clusters = self._cluster_patches_spatially(patches)

        return AggregatedResult(
            image_name=image_name,
            image_path=patches[0].image_path,
            doc_score=float(doc_score),
            patch_count=len(patches),
            best_patch=best_patch,
            all_patches=sorted(patches, key=lambda p: p.score, reverse=True),
            aggregation_method=self.aggregation_method,
            spatial_clusters=spatial_clusters,
        )

    def _cluster_patches_spatially(self, patches: list[PatchResult]) -> list[list[PatchResult]]:
        """Cluster patches that are spatially close to each other."""
        if len(patches) <= 1:
            return [patches]

        clusters = []
        remaining_patches = patches.copy()

        while remaining_patches:
            # Start new cluster with highest scoring remaining patch
            seed_patch = max(remaining_patches, key=lambda p: p.score)
            current_cluster = [seed_patch]
            remaining_patches.remove(seed_patch)

            # Add nearby patches to cluster
            added_to_cluster = True
            while added_to_cluster:
                added_to_cluster = False
                for patch in remaining_patches.copy():
                    if self._is_patch_nearby(patch, current_cluster):
                        current_cluster.append(patch)
                        remaining_patches.remove(patch)
                        added_to_cluster = True

            clusters.append(current_cluster)

        return sorted(clusters, key=lambda cluster: max(p.score for p in cluster), reverse=True)

    def _is_patch_nearby(self, patch: PatchResult, cluster: list[PatchResult]) -> bool:
        """Check if a patch is spatially close to any patch in the cluster."""
        patch_center = self._get_patch_center(patch.coordinates)

        for cluster_patch in cluster:
            cluster_center = self._get_patch_center(cluster_patch.coordinates)
            distance = np.sqrt(
                (patch_center[0] - cluster_center[0]) ** 2
                + (patch_center[1] - cluster_center[1]) ** 2
            )

            if distance <= self.cluster_distance_threshold:
                return True

        return False

    def _get_patch_center(self, coordinates: tuple[int, int, int, int]) -> tuple[float, float]:
        """Get center point of a patch."""
        x1, y1, x2, y2 = coordinates
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def print_aggregated_results(
        self, results: list[AggregatedResult], max_patches_per_doc: int = 3
    ):
        """Pretty print aggregated results."""
        print(f"\n🔍 Aggregated Results (method: {self.aggregation_method})")
        print("=" * 80)

        for i, result in enumerate(results):
            print(f"\n{i + 1}. {result.image_name}")
            print(f"   Doc Score: {result.doc_score:.4f} | Patches: {result.patch_count}")
            print(f"   Path: {result.image_path}")

            # Show best patch
            best = result.best_patch
            print(
                f"   🌟 Best Patch: #{best.patch_id} at {best.coordinates} (score: {best.score:.4f})"
            )

            # Show top patches
            print("   📍 Top Patches:")
            for j, patch in enumerate(result.all_patches[:max_patches_per_doc]):
                print(
                    f"      {j + 1}. Patch #{patch.patch_id}: {patch.score:.4f} at {patch.coordinates}"
                )

            # Show spatial clusters if available
            if result.spatial_clusters and len(result.spatial_clusters) > 1:
                print(f"   🗂️ Spatial Clusters: {len(result.spatial_clusters)}")
                for j, cluster in enumerate(result.spatial_clusters[:2]):  # Show top 2 clusters
                    cluster_score = max(p.score for p in cluster)
                    print(
                        f"      Cluster {j + 1}: {len(cluster)} patches (best: {cluster_score:.4f})"
                    )


def demo_aggregation():
    """Demonstrate the multi-vector aggregation functionality."""
    print("=== Multi-Vector Aggregation Demo ===")

    # Simulate some patch-level search results
    # In real usage, these would come from LeannSearcher.search()

    class MockResult:
        def __init__(self, score, metadata):
            self.score = score
            self.metadata = metadata

    # Simulate results for 2 images with multiple patches each
    mock_results = [
        # Image 1: cats_and_kitchen.jpg - 4 patches
        MockResult(
            0.85,
            {
                "image_name": "cats_and_kitchen.jpg",
                "image_path": "/path/to/cats_and_kitchen.jpg",
                "patch_id": 3,
                "coordinates": [100, 50, 224, 174],  # Kitchen area
                "attention_score": 0.92,
                "scale": 1.0,
            },
        ),
        MockResult(
            0.78,
            {
                "image_name": "cats_and_kitchen.jpg",
                "image_path": "/path/to/cats_and_kitchen.jpg",
                "patch_id": 7,
                "coordinates": [200, 300, 324, 424],  # Cat area
                "attention_score": 0.88,
                "scale": 1.0,
            },
        ),
        MockResult(
            0.72,
            {
                "image_name": "cats_and_kitchen.jpg",
                "image_path": "/path/to/cats_and_kitchen.jpg",
                "patch_id": 12,
                "coordinates": [150, 100, 274, 224],  # Appliances
                "attention_score": 0.75,
                "scale": 1.0,
            },
        ),
        MockResult(
            0.65,
            {
                "image_name": "cats_and_kitchen.jpg",
                "image_path": "/path/to/cats_and_kitchen.jpg",
                "patch_id": 15,
                "coordinates": [50, 250, 174, 374],  # Furniture
                "attention_score": 0.70,
                "scale": 1.0,
            },
        ),
        # Image 2: city_street.jpg - 3 patches
        MockResult(
            0.68,
            {
                "image_name": "city_street.jpg",
                "image_path": "/path/to/city_street.jpg",
                "patch_id": 2,
                "coordinates": [300, 100, 424, 224],  # Buildings
                "attention_score": 0.80,
                "scale": 1.0,
            },
        ),
        MockResult(
            0.62,
            {
                "image_name": "city_street.jpg",
                "image_path": "/path/to/city_street.jpg",
                "patch_id": 8,
                "coordinates": [100, 350, 224, 474],  # Street level
                "attention_score": 0.75,
                "scale": 1.0,
            },
        ),
        MockResult(
            0.55,
            {
                "image_name": "city_street.jpg",
                "image_path": "/path/to/city_street.jpg",
                "patch_id": 11,
                "coordinates": [400, 200, 524, 324],  # Sky area
                "attention_score": 0.60,
                "scale": 1.0,
            },
        ),
    ]

    # Test different aggregation methods
    methods = ["maxsim", "voting", "weighted", "mean"]

    for method in methods:
        print(f"\n{'=' * 20} {method.upper()} AGGREGATION {'=' * 20}")

        aggregator = MultiVectorAggregator(
            aggregation_method=method,
            spatial_clustering=True,
            cluster_distance_threshold=100.0,
        )

        aggregated = aggregator.aggregate_results(mock_results, top_k=5)
        aggregator.print_aggregated_results(aggregated)


if __name__ == "__main__":
    demo_aggregation()
