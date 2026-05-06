"""
Shi et al. early burst detection + ASMK image retrieval pipeline,
extended with spatial reranking, average query expansion (AQE),
and discriminative query expansion (DQE) following Arandjelovic & Zisserman 2012.
"""

from __future__ import annotations

import math
import hashlib
import importlib
import os
import shutil
import sys
import tarfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOCAL_ASMK_ROOT = PROJECT_ROOT / "asmk"
CONTENT_ROOT = "./content"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


try:
    _notify_module = importlib.import_module("inference_helpers")
    _notify: Callable[[str], None] | None = _notify_module.notify
except ModuleNotFoundError:
    _notify = print


def notify_event(message: str) -> None:
    if _notify is None:
        print(f"[notify unavailable] {message}")
        return
    _notify(message)


"""
1. Dataset processing
"""


@dataclass(frozen=True)
class DatasetConfig:
    archive_dir: Path = Path(f"{CONTENT_ROOT}/oxbuildings")
    extracted_root: Path = Path(f"{CONTENT_ROOT}/oxbuildings_extracted")
    dataset_root: Path = Path(f"{CONTENT_ROOT}/dataset_unificado")
    metadata_csv: Path = Path(f"{CONTENT_ROOT}/dataset_unificado/metadata.csv")
    copy_files: bool = True
    kaggle_dataset: str = "skylord/oxbuildings"

    @property
    def oxford_source(self) -> Path:
        return self.extracted_root / "oxbuild_images"

    @property
    def paris_sources(self) -> tuple[Path, Path]:
        return (
            self.extracted_root / "paris_1" / "paris",
            self.extracted_root / "paris_2" / "paris",
        )


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def oxford_class_from_name(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer Oxford class from filename: {filename}")
    return "_".join(parts[:-1])


def make_unique_path(dst_file: Path) -> Path:
    if not dst_file.exists():
        return dst_file

    parent = dst_file.parent
    for suffix_id in range(1, 1_000_000):
        candidate = parent / f"{dst_file.stem}_{suffix_id}{dst_file.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique path for {dst_file}")


def copy_or_move(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(src, dst)
        return
    shutil.move(str(src), str(dst))


def safe_extract_tar(tar: tarfile.TarFile, dst_dir: Path) -> None:
    dst_dir_resolved = dst_dir.resolve()
    for member in tar.getmembers():
        member_path = (dst_dir / member.name).resolve()
        try:
            member_path.relative_to(dst_dir_resolved)
        except ValueError:
            raise RuntimeError(f"Blocked unsafe tar member path: {member.name}")
    tar.extractall(dst_dir)


def extract_archives(config: DatasetConfig) -> None:
    if not config.archive_dir.exists():
        raise FileNotFoundError(f"Archive directory does not exist: {config.archive_dir}")

    config.extracted_root.mkdir(parents=True, exist_ok=True)
    for archive_path in sorted(config.archive_dir.iterdir()):
        if archive_path.suffix != ".tgz":
            continue
        output_dir = config.extracted_root / archive_path.name.replace(".tgz", "")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive_path.name} -> {output_dir}")
        with tarfile.open(archive_path, "r:gz") as tar:
            safe_extract_tar(tar, output_dir)


def download_kaggle_dataset(config: DatasetConfig) -> None:
    import kagglehub

    config.archive_dir.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download(config.kaggle_dataset, output_dir=str(config.archive_dir))


def build_unified_dataset(config: DatasetConfig) -> pd.DataFrame:
    config.dataset_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    if not config.oxford_source.exists():
        raise FileNotFoundError(f"Oxford image folder does not exist: {config.oxford_source}")

    for image_path in sorted(config.oxford_source.rglob("*")):
        if not image_path.is_file() or not is_image_file(image_path):
            continue
        class_name = oxford_class_from_name(image_path.name)
        dst_file = make_unique_path(config.dataset_root / "oxford" / class_name / image_path.name)
        copy_or_move(image_path, dst_file, copy_files=config.copy_files)
        records.append(
            {
                "dataset": "oxford",
                "class_name": class_name,
                "img_name": dst_file.name,
                "img_path": str(dst_file),
                "source_path": str(image_path),
            }
        )

    for paris_source in config.paris_sources:
        if not paris_source.exists():
            raise FileNotFoundError(f"Paris image folder does not exist: {paris_source}")
        for class_dir in sorted(paris_source.iterdir()):
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or not is_image_file(image_path):
                    continue
                dst_file = make_unique_path(
                    config.dataset_root / "paris" / class_dir.name / image_path.name
                )
                copy_or_move(image_path, dst_file, copy_files=config.copy_files)
                records.append(
                    {
                        "dataset": "paris",
                        "class_name": class_dir.name,
                        "img_name": dst_file.name,
                        "img_path": str(dst_file),
                        "source_path": str(image_path),
                    }
                )

    metadata = pd.DataFrame(records)
    if metadata.empty:
        raise RuntimeError("No Oxford/Paris images were found after dataset unification.")
    metadata.to_csv(config.metadata_csv, index=False)
    return metadata


def load_or_build_metadata(config: DatasetConfig, rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not config.metadata_csv.exists():
        if not config.archive_dir.exists() or not any(config.archive_dir.iterdir()):
            download_kaggle_dataset(config)
        if config.archive_dir.exists():
            extract_archives(config)
        return build_unified_dataset(config)

    metadata = pd.read_csv(config.metadata_csv)
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata: pd.DataFrame) -> None:
    required_columns = {"dataset", "class_name", "img_name", "img_path"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    if metadata.empty:
        raise ValueError("Metadata is empty.")


"""
2. Feature extraction
"""


FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
IntVector = NDArray[np.int64]


@dataclass(frozen=True)
class FeatureConfig:
    max_features_per_image: int | None = None
    contrast_threshold: float = 0.01
    edge_threshold: float = 10.0
    cache_dir: Path = Path(f"{CONTENT_ROOT}/cache/features")
    detector: Literal["hesaff", "sift"] = "hesaff"


@dataclass(frozen=True)
class FeatureSet:
    descriptors: FloatMatrix
    positions: FloatMatrix
    scales: FloatVector
    orientations: FloatVector
    responses: FloatVector


def create_sift(config: FeatureConfig) -> cv2.SIFT:
    return cv2.SIFT_create(
        nfeatures=0 if config.max_features_per_image is None else config.max_features_per_image,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
    )


def l2_normalize_rows(vecs: FloatMatrix, eps: float = 1e-12) -> FloatMatrix:
    norms = np.linalg.norm(vecs, ord=2, axis=1, keepdims=True)
    return (vecs / np.maximum(norms, eps)).astype(np.float32)


def rootsift(sift_descriptors: FloatMatrix, eps: float = 1e-12) -> FloatMatrix:
    descriptors = sift_descriptors.astype(np.float32)
    l1_norm = descriptors.sum(axis=1, keepdims=True)
    descriptors = descriptors / np.maximum(l1_norm, eps)
    descriptors = np.sqrt(descriptors)
    return l2_normalize_rows(descriptors)


def empty_feature_set(dim: int = 128) -> FeatureSet:
    return FeatureSet(
        descriptors=np.empty((0, dim), dtype=np.float32),
        positions=np.empty((0, 2), dtype=np.float32),
        scales=np.empty((0,), dtype=np.float32),
        orientations=np.empty((0,), dtype=np.float32),
        responses=np.empty((0,), dtype=np.float32),
    )


def feature_cache_path(image_path: Path, config: FeatureConfig) -> Path:
    safe_name = str(image_path).strip("/").replace("/", "__")
    feature_limit = "all" if config.max_features_per_image is None else f"max{config.max_features_per_image}"
    cache_name = f"{feature_limit}_{safe_name}.npz"
    return config.cache_dir / config.detector / cache_name


def extract_sift_rootsift(image_path: Path, sift: cv2.SIFT) -> FeatureSet:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None or len(keypoints) == 0:
        return empty_feature_set()

    root_descriptors = rootsift(descriptors)
    positions = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    scales = np.array([kp.size for kp in keypoints], dtype=np.float32)
    orientations = np.deg2rad(np.array([kp.angle if kp.angle >= 0 else 0.0 for kp in keypoints]))
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    return FeatureSet(root_descriptors, positions, scales, orientations.astype(np.float32), responses)


def extract_hesaff_rootsift(image_path: Path, config: FeatureConfig) -> FeatureSet:
    try:
        pyhesaff = importlib.import_module("pyhesaff")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "FeatureConfig.detector='hesaff' requires pyhesaff. "
            "Install it or set FeatureConfig(detector='sift')."
        ) from error
    try:
        result = pyhesaff.detect_feats(str(image_path))
    except Exception as error:
        print(f"[WARN] pyhesaff failed to detect features: {error}")
        return empty_feature_set()

    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError(f"pyhesaff returned an invalid feature tuple: {type(result)}")
    keypoints, descriptors = result[:2]

    if descriptors is None or keypoints is None or len(keypoints) == 0:
        return empty_feature_set()

    keypoints = np.asarray(keypoints, dtype=np.float32)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] != 128:
        raise ValueError(f"pyhesaff returned descriptors with invalid shape: {descriptors.shape}")
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError(f"pyhesaff returned keypoints with invalid shape: {keypoints.shape}")

    positions = keypoints[:, :2].astype(np.float32)
    if keypoints.shape[1] >= 5:
        a11 = keypoints[:, 2]
        a21 = keypoints[:, 3]
        a22 = keypoints[:, 4]
        determinant = np.maximum(a11 * a22 - np.square(a21), 1e-12)
        scales = np.sqrt(1.0 / np.sqrt(determinant)).astype(np.float32)
    else:
        scales = np.ones((keypoints.shape[0],), dtype=np.float32)
    if keypoints.shape[1] >= 6:
        orientations = keypoints[:, 5].astype(np.float32)
    else:
        orientations = np.zeros((keypoints.shape[0],), dtype=np.float32)
    responses = scales.astype(np.float32)
    return FeatureSet(rootsift(descriptors), positions, scales, orientations, responses)


def extract_features(image_path: Path, sift: cv2.SIFT | None, config: FeatureConfig) -> FeatureSet:
    if config.detector == "hesaff":
        features = extract_hesaff_rootsift(image_path, config)
    elif config.detector == "sift":
        if sift is None:
            raise ValueError("SIFT detector object is required when FeatureConfig.detector='sift'.")
        features = extract_sift_rootsift(image_path, sift)
    else:
        raise ValueError(f"Unknown feature detector: {config.detector}")

    if config.max_features_per_image is None:
        return features
    if features.descriptors.shape[0] <= config.max_features_per_image:
        return features
    return FeatureSet(
        descriptors=features.descriptors,
        positions=features.positions,
        scales=features.scales,
        orientations=features.orientations,
        responses=features.responses,
    )


def load_or_extract_rootsift(image_path: Path, sift: cv2.SIFT | None, config: FeatureConfig) -> FeatureSet:
    cache_path = feature_cache_path(image_path, config)
    if cache_path.exists():
        cached = np.load(cache_path)
        return FeatureSet(
            descriptors=cached["descriptors"].astype(np.float32),
            positions=cached["positions"].astype(np.float32),
            scales=cached["scales"].astype(np.float32),
            orientations=cached["orientations"].astype(np.float32),
            responses=cached["responses"].astype(np.float32),
        )
    try:
        features = extract_features(image_path, sift, config)
    except ValueError as error:
        features = empty_feature_set()
        print(f"[WARN] {error}")
        return features
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        descriptors=features.descriptors,
        positions=features.positions,
        scales=features.scales,
        orientations=features.orientations,
        responses=features.responses,
    )
    return features


"""
3. Early burst detection and aggregation
"""


@dataclass(frozen=True)
class BurstConfig:
    tau: float | None = None
    descriptor_sigmoid_b: float = 0.78
    descriptor_sigmoid_w: float = 35.0
    use_scale_kernel: bool = True
    use_orientation_kernel: bool = True
    scale_lambda: float = 1.0
    orientation_kappa: float = 4.0
    max_pairwise_features: int | None = None
    cache_dir: Path = Path(f"{CONTENT_ROOT}/cache/bursts")


def burst_cache_path(
    image_path: Path,
    feature_config: FeatureConfig,
    burst_config: BurstConfig,
    tau: float | None,
) -> Path:
    safe_name = str(image_path).strip("/").replace("/", "__")
    feature_limit = (
        "all" if feature_config.max_features_per_image is None else f"max{feature_config.max_features_per_image}"
    )
    signature_parts = (
        "v1",
        feature_config.detector,
        feature_limit,
        feature_config.contrast_threshold,
        feature_config.edge_threshold,
        tau,
        burst_config.descriptor_sigmoid_b,
        burst_config.descriptor_sigmoid_w,
        burst_config.use_scale_kernel,
        burst_config.use_orientation_kernel,
        burst_config.scale_lambda,
        burst_config.orientation_kappa,
        burst_config.max_pairwise_features,
    )
    signature = hashlib.sha1(repr(signature_parts).encode("utf-8")).hexdigest()[:16]
    return burst_config.cache_dir / feature_config.detector / f"{signature}_{safe_name}.npz"


def load_cached_burst_descriptors(cache_path: Path) -> tuple[FloatMatrix, int] | None:
    if not cache_path.exists():
        return None

    with np.load(cache_path) as cached:
        return cached["descriptors"].astype(np.float32), int(cached["n_features"])


def save_cached_burst_descriptors(cache_path: Path, descriptors: FloatMatrix, n_features: int) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        descriptors=descriptors.astype(np.float32),
        n_features=np.array(n_features, dtype=np.int64),
    )


def keep_strongest_features(features: FeatureSet, max_features: int | None) -> FeatureSet:
    if max_features is None:
        return features
    if features.descriptors.shape[0] <= max_features:
        return features

    strongest = np.argsort(-features.responses)[:max_features]
    strongest.sort()
    return FeatureSet(
        descriptors=features.descriptors[strongest],
        positions=features.positions[strongest],
        scales=features.scales[strongest],
        orientations=features.orientations[strongest],
        responses=features.responses[strongest],
    )


def validate_burst_config(config: BurstConfig) -> None:
    if config.tau is not None and not 0.0 <= config.tau <= 1.0:
        raise ValueError(f"config.tau must be in [0, 1], got {config.tau}")
    if config.descriptor_sigmoid_w <= 0:
        raise ValueError(
            f"config.descriptor_sigmoid_w must be positive, got {config.descriptor_sigmoid_w}"
        )


def descriptor_kernel(inner_products: FloatMatrix, b: float, w: float) -> FloatMatrix:
    logits = np.clip(w * (inner_products.astype(np.float32) - b), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def scale_affinity(scales: FloatVector, scale_lambda: float) -> FloatMatrix:
    safe_scales = np.maximum(scales.astype(np.float32), 1e-6)
    log_ratio = np.log(safe_scales[:, None] / safe_scales[None, :])
    return np.exp(-scale_lambda * np.square(log_ratio)).astype(np.float32)


def orientation_affinity(orientations: FloatVector, kappa: float) -> FloatMatrix:
    if kappa <= 0:
        return np.ones((orientations.shape[0], orientations.shape[0]), dtype=np.float32)

    delta = orientations[:, None] - orientations[None, :]
    numerator = np.exp(kappa * np.cos(delta)) - np.exp(-kappa)
    denominator = 2.0 * math.sinh(kappa)
    return (numerator / denominator).astype(np.float32)


def burst_affinity_matrix(features: FeatureSet, config: BurstConfig) -> FloatMatrix:
    validate_burst_config(config)
    descriptors = l2_normalize_rows(features.descriptors)
    inner_products = np.clip(descriptors @ descriptors.T, 0.0, 1.0).astype(np.float32)
    affinity = descriptor_kernel(
        inner_products,
        b=config.descriptor_sigmoid_b,
        w=config.descriptor_sigmoid_w,
    )

    if config.use_scale_kernel:
        affinity *= scale_affinity(features.scales, config.scale_lambda)
    if config.use_orientation_kernel:
        affinity *= orientation_affinity(features.orientations, config.orientation_kappa)

    np.fill_diagonal(affinity, 1.0)
    return affinity


def burst_tau(config: BurstConfig, fallback_tau: float) -> float:
    tau = fallback_tau if config.tau is None else config.tau
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"Burst threshold tau must be in [0, 1], got {tau}")
    return tau


def aggregate_bursts(features: FeatureSet, config: BurstConfig, tau: float | None = None) -> FloatMatrix:
    if features.descriptors.shape[0] == 0:
        return features.descriptors

    bounded_features = keep_strongest_features(features, config.max_pairwise_features)
    if bounded_features.descriptors.shape[0] == 1:
        return bounded_features.descriptors

    affinity = burst_affinity_matrix(bounded_features, config)
    adjacency = sparse.csr_matrix(affinity >= burst_tau(config, fallback_tau=0.5 if tau is None else tau))
    _n_components, labels = connected_components(adjacency, directed=False)

    aggregated: list[FloatVector] = []
    for label in np.unique(labels):
        group = bounded_features.descriptors[labels == label]
        descriptor = group.mean(axis=0, dtype=np.float32)
        aggregated.append(descriptor.astype(np.float32))

    return l2_normalize_rows(np.vstack(aggregated).astype(np.float32))


"""
4. ASMK vocabulary generation and image representation
"""


@dataclass(frozen=True)
class ASMKConfig:
    codebook_size: int = 65_536
    gpu_id: int | None = None
    binary: bool = True
    use_idf: bool = True
    db_multiple_assignment: int = 1
    query_multiple_assignment: int = 5
    similarity_threshold: float = 0.0
    alpha: float = 3.0
    topk: int | None = None
    train_sample_size: int = 250_000
    cache_dir: Path = Path(f"{CONTENT_ROOT}/cache/asmk")


@dataclass(frozen=True)
class CalibrationConfig:
    sample_size: int = 100
    seed: int = 0
    target_aggregation: float = 0.85
    tau_grid: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99)


@dataclass(frozen=True)
class GroundTruthConfig:
    oxford_gt_dir: Path = Path(f"{CONTENT_ROOT}/oxford_gt_files")
    paris_gt_dir: Path = Path(f"{CONTENT_ROOT}/paris_gt_files")


@dataclass(frozen=True)
class SpatialRerankConfig:
    enabled: bool = True
    top_n: int = 200
    ratio_test: float = 0.8
    ransac_threshold: float = 8.0
    ransac_max_iters: int = 500
    ransac_confidence: float = 0.99
    min_inliers: int = 4
    max_query_descriptors: int = 1500
    max_candidate_descriptors: int = 3000


@dataclass(frozen=True)
class QueryExpansionConfig:
    enable_aqe: bool = True
    enable_dqe: bool = True
    n_positive: int = 10
    n_negative: int = 200
    svm_c: float = 1.0
    svm_max_iter: int = 2000
    quantize_chunk_size: int = 4000


@dataclass(frozen=True)
class DescriptorTable:
    descriptors: FloatMatrix
    image_ids: IntVector


@dataclass(frozen=True)
class ImageDescriptorTask:
    dataset: str
    class_name: str
    img_name: str
    img_path: str


@dataclass(frozen=True)
class ImageDescriptorResult:
    task: ImageDescriptorTask
    db_descriptors: FloatMatrix
    query_descriptors: FloatMatrix
    n_features: int = 0
    n_aggregated: int = 0
    skip_reason: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    burst: BurstConfig = field(default_factory=BurstConfig)
    asmk: ASMKConfig = field(default_factory=ASMKConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)
    spatial_rerank: SpatialRerankConfig = field(default_factory=SpatialRerankConfig)
    query_expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)
    output_dir: Path = Path(f"{CONTENT_ROOT}/shi_asmk_outputs")
    rebuild_metadata: bool = False
    aggregate_queries: bool = False
    random_seed: int = 0
    results_topk: int = 50
    descriptor_workers: int | None = None
    descriptor_chunksize: int = 4


def import_asmk_method() -> type:
    if LOCAL_ASMK_ROOT.exists() and str(LOCAL_ASMK_ROOT) not in sys.path:
        sys.path.insert(0, str(LOCAL_ASMK_ROOT))
    try:
        from asmk import ASMKMethod
    except ImportError as error:
        raise ImportError(
            "Could not import local ASMK. Install dependencies and the local package with "
            "`uv sync` or the command in pip.txt before running this pipeline."
        ) from error
    return ASMKMethod


def stack_or_empty(chunks: list[FloatMatrix], dim: int = 128) -> FloatMatrix:
    if not chunks:
        return np.empty((0, dim), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32)


def concatenate_ids(chunks: list[IntVector]) -> IntVector:
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64)


def resolve_descriptor_workers(requested_workers: int | None) -> int:
    if requested_workers is not None:
        if requested_workers < 1:
            raise ValueError(f"descriptor_workers must be >= 1, got {requested_workers}")
        return requested_workers

    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, min(cpu_count - 1, 8))


def metadata_to_descriptor_tasks(metadata: pd.DataFrame) -> list[ImageDescriptorTask]:
    return [
        ImageDescriptorTask(
            dataset=str(row.dataset),
            class_name=str(row.class_name),
            img_name=str(row.img_name),
            img_path=str(row.img_path),
        )
        for row in metadata.itertuples(index=False)
    ]


def process_image_descriptors(
    args: tuple[ImageDescriptorTask, FeatureConfig, BurstConfig, bool, float],
) -> ImageDescriptorResult:
    task, feature_config, burst_config, aggregate_queries, tau = args
    cv2.setNumThreads(1)
    sift = create_sift(feature_config) if feature_config.detector == "sift" else None
    image_path = Path(task.img_path)
    burst_cache = load_cached_burst_descriptors(
        burst_cache_path(image_path, feature_config, burst_config, tau)
    )
    if burst_cache is not None:
        db_descriptors, n_features = burst_cache
        if db_descriptors.shape[0] == 0:
            return ImageDescriptorResult(
                task=task,
                db_descriptors=db_descriptors,
                query_descriptors=empty_feature_set().descriptors,
                n_features=n_features,
                n_aggregated=0,
                skip_reason="empty descriptor aggregation",
            )
        return ImageDescriptorResult(
            task=task,
            db_descriptors=db_descriptors,
            query_descriptors=empty_feature_set().descriptors,
            n_features=n_features,
            n_aggregated=db_descriptors.shape[0],
        )

    features = load_or_extract_rootsift(image_path, sift, feature_config)
    if features.descriptors.shape[0] == 0:
        return ImageDescriptorResult(
            task=task,
            db_descriptors=empty_feature_set().descriptors,
            query_descriptors=empty_feature_set().descriptors,
            skip_reason="no valid features",
        )

    db_descriptors = aggregate_bursts(features, burst_config, tau=tau)
    save_cached_burst_descriptors(
        burst_cache_path(image_path, feature_config, burst_config, tau),
        db_descriptors,
        features.descriptors.shape[0],
    )
    if db_descriptors.shape[0] == 0:
        return ImageDescriptorResult(
            task=task,
            db_descriptors=db_descriptors,
            query_descriptors=empty_feature_set().descriptors,
            n_features=features.descriptors.shape[0],
            n_aggregated=db_descriptors.shape[0],
            skip_reason="empty descriptor aggregation",
        )

    return ImageDescriptorResult(
        task=task,
        db_descriptors=db_descriptors,
        query_descriptors=empty_feature_set().descriptors,
        n_features=features.descriptors.shape[0],
        n_aggregated=db_descriptors.shape[0],
    )


def iter_descriptor_results(
    tasks: list[ImageDescriptorTask],
    feature_config: FeatureConfig,
    burst_config: BurstConfig,
    aggregate_queries: bool,
    tau_by_dataset: dict[str, float],
    max_workers: int,
    chunksize: int,
) -> Iterator[ImageDescriptorResult]:
    if chunksize < 1:
        raise ValueError(f"descriptor_chunksize must be >= 1, got {chunksize}")

    worker_args = (
        (task, feature_config, burst_config, aggregate_queries, tau_by_dataset[task.dataset])
        for task in tasks
    )
    if max_workers == 1:
        for args in tqdm(
            worker_args,
            total=len(tasks),
            desc="Extract RootSIFT + Shi bursts",
        ):
            yield process_image_descriptors(args)
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        yield from tqdm(
            executor.map(process_image_descriptors, worker_args, chunksize=chunksize),
            total=len(tasks),
            desc=f"Extract RootSIFT + Shi bursts ({max_workers} workers)",
        )


def build_descriptor_tables(
    metadata: pd.DataFrame,
    feature_config: FeatureConfig,
    burst_config: BurstConfig,
    aggregate_queries: bool,
    tau_by_dataset: dict[str, float],
    descriptor_workers: int | None = None,
    descriptor_chunksize: int = 4,
) -> tuple[pd.DataFrame, DescriptorTable, DescriptorTable]:
    validate_metadata(metadata)
    max_workers = resolve_descriptor_workers(descriptor_workers)

    valid_records: list[dict[str, object]] = []
    db_descriptor_chunks: list[FloatMatrix] = []
    db_id_chunks: list[IntVector] = []
    query_descriptor_chunks: list[FloatMatrix] = []
    query_id_chunks: list[IntVector] = []

    tasks = metadata_to_descriptor_tasks(metadata)
    results = iter_descriptor_results(
        tasks,
        feature_config,
        burst_config,
        aggregate_queries,
        tau_by_dataset,
        max_workers=max_workers,
        chunksize=descriptor_chunksize,
    )
    for result in results:
        if result.skip_reason is not None:
            print(f"[WARN] Skipping {result.task.img_path}: {result.skip_reason}")
            continue

        image_id = len(valid_records)
        valid_records.append(
            {
                "image_id": image_id,
                "dataset": result.task.dataset,
                "class_name": result.task.class_name,
                "img_name": result.task.img_name,
                "img_path": result.task.img_path,
                "n_features": result.n_features,
                "n_aggregated": result.n_aggregated,
                "aggregation_ratio": result.n_aggregated / result.n_features,
            }
        )
        db_descriptor_chunks.append(result.db_descriptors)
        db_id_chunks.append(np.full(result.db_descriptors.shape[0], image_id, dtype=np.int64))
        if result.query_descriptors.shape[0] > 0:
            query_descriptor_chunks.append(result.query_descriptors)
            query_id_chunks.append(np.full(result.query_descriptors.shape[0], image_id, dtype=np.int64))

    valid_metadata = pd.DataFrame(valid_records)
    if valid_metadata.empty:
        raise RuntimeError("No valid images remained after feature extraction.")

    db_table = DescriptorTable(stack_or_empty(db_descriptor_chunks), concatenate_ids(db_id_chunks))
    query_table = DescriptorTable(stack_or_empty(query_descriptor_chunks), concatenate_ids(query_id_chunks))
    return valid_metadata, db_table, query_table


def calibrate_tau_for_dataset(
    metadata: pd.DataFrame,
    dataset_name: str,
    feature_config: FeatureConfig,
    burst_config: BurstConfig,
    calibration_config: CalibrationConfig,
) -> pd.DataFrame:
    dataset_metadata = metadata[metadata["dataset"] == dataset_name].reset_index(drop=True)
    if dataset_metadata.empty:
        raise ValueError(f"Cannot calibrate tau for empty dataset: {dataset_name}")

    sample_size = min(calibration_config.sample_size, len(dataset_metadata))
    sample = dataset_metadata.sample(
        n=sample_size,
        random_state=calibration_config.seed,
        replace=False,
    )
    sift = create_sift(feature_config) if feature_config.detector == "sift" else None
    features_per_image: list[FeatureSet] = []
    for row in tqdm(sample.itertuples(index=False), total=sample_size, desc=f"Calibrate {dataset_name}"):
        features = load_or_extract_rootsift(Path(row.img_path), sift, feature_config)
        if features.descriptors.shape[0] > 0:
            features_per_image.append(features)

    if not features_per_image:
        raise RuntimeError(f"No valid features found while calibrating dataset: {dataset_name}")

    rows: list[dict[str, object]] = []
    for tau in calibration_config.tau_grid:
        ratios: list[float] = []
        for features in features_per_image:
            aggregated = aggregate_bursts(features, burst_config, tau=tau)
            ratios.append(aggregated.shape[0] / features.descriptors.shape[0])
        aggregation_ratio = float(np.mean(ratios))
        rows.append(
            {
                "dataset": dataset_name,
                "tau": tau,
                "aggregation_ratio": aggregation_ratio,
                "target_aggregation": calibration_config.target_aggregation,
                "absolute_error": abs(aggregation_ratio - calibration_config.target_aggregation),
            }
        )

    calibration = pd.DataFrame(rows)
    best_idx = calibration["absolute_error"].idxmin()
    calibration["chosen"] = False
    calibration.loc[best_idx, "chosen"] = True
    return calibration


def calibrate_taus(
    metadata: pd.DataFrame,
    feature_config: FeatureConfig,
    burst_config: BurstConfig,
    calibration_config: CalibrationConfig,
    output_dir: Path,
) -> dict[str, float]:
    if burst_config.tau is not None:
        return {
            str(dataset): burst_config.tau
            for dataset in sorted(metadata["dataset"].unique())
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    tau_by_dataset: dict[str, float] = {}
    for dataset in sorted(metadata["dataset"].unique()):
        calibration = calibrate_tau_for_dataset(
            metadata,
            dataset_name=str(dataset),
            feature_config=feature_config,
            burst_config=burst_config,
            calibration_config=calibration_config,
        )
        calibration.to_csv(output_dir / f"calibration_{dataset}.csv", index=False)
        chosen = calibration[calibration["chosen"]].iloc[0]
        tau_by_dataset[str(dataset)] = float(chosen["tau"])
        print(
            f"Calibrated {dataset}: tau={chosen['tau']:.3f}, "
            f"aggregation%={chosen['aggregation_ratio']:.3f}"
        )
    return tau_by_dataset


def sample_training_descriptors(descriptors: FloatMatrix, config: PipelineConfig) -> FloatMatrix:
    if descriptors.shape[0] < config.asmk.codebook_size:
        raise ValueError(
            f"Need at least {config.asmk.codebook_size} descriptors to train the codebook; "
            f"got {descriptors.shape[0]}."
        )

    sample_size = min(config.asmk.train_sample_size, descriptors.shape[0])
    rng = np.random.default_rng(config.random_seed)
    sample_ids = rng.choice(descriptors.shape[0], size=sample_size, replace=False)
    return np.ascontiguousarray(descriptors[sample_ids], dtype=np.float32)


def asmk_params(config: ASMKConfig) -> dict[str, object]:
    return {
        "index": {"gpu_id": config.gpu_id},
        "train_codebook": {"codebook": {"size": config.codebook_size}},
        "build_ivf": {
            "kernel": {"binary": config.binary},
            "ivf": {"use_idf": config.use_idf},
            "quantize": {"multiple_assignment": config.db_multiple_assignment},
            "aggregate": {},
        },
        "query_ivf": {
            "quantize": {"multiple_assignment": config.query_multiple_assignment},
            "aggregate": {},
            "search": {"topk": config.topk},
            "similarity": {
                "similarity_threshold": config.similarity_threshold,
                "alpha": config.alpha,
            },
        },
    }


def train_and_index_asmk(
    db_table: DescriptorTable,
    config: PipelineConfig,
    dataset_name: str,
) -> object:
    ASMKMethod = import_asmk_method()
    config.asmk.cache_dir.mkdir(parents=True, exist_ok=True)
    binary_label = "bin" if config.asmk.binary else "nobin"
    codebook_path = config.asmk.cache_dir / f"codebook_k{config.asmk.codebook_size}_{dataset_name}.pkl"
    ivf_path = config.asmk.cache_dir / (
        f"ivf_k{config.asmk.codebook_size}_{binary_label}_idf{int(config.asmk.use_idf)}_{dataset_name}.pkl"
    )

    train_descriptors = sample_training_descriptors(db_table.descriptors, config)
    asmk = ASMKMethod.initialize_untrained(asmk_params(config.asmk))
    asmk = asmk.train_codebook(train_descriptors, cache_path=str(codebook_path))
    ivf_builder = asmk.create_ivf_builder(cache_path=str(ivf_path))
    if not ivf_builder.loaded_from_cache:
        ivf_builder.add(db_table.descriptors, db_table.image_ids, progress=500)
    return asmk.add_ivf_builder(ivf_builder)


"""
5. Ground truth and query construction
"""


@dataclass(frozen=True)
class GroundTruthQuery:
    dataset: str
    landmark: str
    query_name: str
    query_basename: str
    bbox: tuple[float, float, float, float]
    positive_ids: set[int]
    junk_ids: set[int]


def image_basename(path_or_name: str) -> str:
    return Path(path_or_name).stem


def strip_query_prefix(name: str) -> str:
    basename = image_basename(name)
    for prefix in ("oxc1_",):
        if basename.startswith(prefix):
            return basename[len(prefix):]
    return basename


def read_gt_names(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {path}")
    names: set[str] = set()
    for line in path.read_text().splitlines():
        value = line.strip()
        if value:
            names.add(image_basename(value))
    return names


def metadata_basename_index(metadata: pd.DataFrame, dataset_name: str) -> dict[str, int]:
    dataset_metadata = metadata[metadata["dataset"] == dataset_name]
    index: dict[str, int] = {}
    for row in dataset_metadata.itertuples(index=False):
        basename = image_basename(row.img_name)
        image_id = int(row.image_id)
        if basename in index:
            raise ValueError(f"Duplicate image basename in {dataset_name}: {basename}")
        index[basename] = image_id
    return index


def gt_dir_for_dataset(config: GroundTruthConfig, dataset_name: str) -> Path:
    if dataset_name == "oxford":
        return config.oxford_gt_dir
    if dataset_name == "paris":
        return config.paris_gt_dir
    raise ValueError(f"No ground-truth directory configured for dataset: {dataset_name}")


def load_ground_truth_for_dataset(
    metadata: pd.DataFrame,
    config: GroundTruthConfig,
    dataset_name: str,
) -> list[GroundTruthQuery]:
    gt_dir = gt_dir_for_dataset(config, dataset_name)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    basename_to_id = metadata_basename_index(metadata, dataset_name)
    queries: list[GroundTruthQuery] = []
    for query_path in sorted(gt_dir.glob("*_query.txt")):
        query_name = query_path.name.removesuffix("_query.txt")
        query_parts = query_path.read_text().split()
        if len(query_parts) != 5:
            raise ValueError(f"Invalid query file format: {query_path}")

        query_basename = strip_query_prefix(query_parts[0])
        if query_basename not in basename_to_id:
            raise KeyError(f"Query image not found in metadata: {query_basename}")
        bbox = tuple(float(value) for value in query_parts[1:5])
        good = read_gt_names(gt_dir / f"{query_name}_good.txt")
        ok = read_gt_names(gt_dir / f"{query_name}_ok.txt")
        junk = read_gt_names(gt_dir / f"{query_name}_junk.txt")
        positives = good.union(ok)

        positive_ids = {
            basename_to_id[basename]
            for basename in positives
            if basename in basename_to_id
        }
        junk_ids = {
            basename_to_id[basename]
            for basename in junk
            if basename in basename_to_id
        }
        queries.append(
            GroundTruthQuery(
                dataset=dataset_name,
                landmark=query_name.rsplit("_", 1)[0],
                query_name=query_name,
                query_basename=query_basename,
                bbox=bbox,
                positive_ids=positive_ids,
                junk_ids=junk_ids,
            )
        )
    if not queries:
        raise RuntimeError(f"No ground-truth queries found in {gt_dir}")
    return queries


def filter_features_by_bbox(features: FeatureSet, bbox: tuple[float, float, float, float]) -> FeatureSet:
    x1, y1, x2, y2 = bbox
    in_bbox = (
        (features.positions[:, 0] >= x1)
        & (features.positions[:, 0] <= x2)
        & (features.positions[:, 1] >= y1)
        & (features.positions[:, 1] <= y2)
    )
    return FeatureSet(
        descriptors=features.descriptors[in_bbox],
        positions=features.positions[in_bbox],
        scales=features.scales[in_bbox],
        orientations=features.orientations[in_bbox],
        responses=features.responses[in_bbox],
    )


@dataclass(frozen=True)
class QueryFeatureBundle:
    raw_features: FeatureSet
    aggregated_descriptors: FloatMatrix


def build_ground_truth_query_bundles(
    metadata: pd.DataFrame,
    queries: list[GroundTruthQuery],
    feature_config: FeatureConfig,
    burst_config: BurstConfig,
    tau: float,
) -> list[QueryFeatureBundle]:
    metadata_by_basename = {
        (row.dataset, image_basename(row.img_name)): row.img_path
        for row in metadata.itertuples(index=False)
    }
    sift = create_sift(feature_config) if feature_config.detector == "sift" else None
    bundles: list[QueryFeatureBundle] = []

    for query in tqdm(queries, desc="Build GT query descriptors"):
        image_path = metadata_by_basename[(query.dataset, query.query_basename)]
        features = load_or_extract_rootsift(Path(image_path), sift, feature_config)
        query_features = filter_features_by_bbox(features, query.bbox)
        if query_features.descriptors.shape[0] == 0:
            print(f"[WARN] GT query has no features in bbox: {query.query_name}")
            bundles.append(QueryFeatureBundle(empty_feature_set(), empty_feature_set().descriptors))
            continue
        aggregated = aggregate_bursts(query_features, burst_config, tau=tau)
        bundles.append(QueryFeatureBundle(query_features, aggregated))

    return bundles


def descriptor_table_from_bundles(bundles: list[QueryFeatureBundle]) -> DescriptorTable:
    chunks: list[FloatMatrix] = []
    ids: list[IntVector] = []
    for query_id, bundle in enumerate(bundles):
        if bundle.aggregated_descriptors.shape[0] == 0:
            continue
        chunks.append(bundle.aggregated_descriptors)
        ids.append(np.full(bundle.aggregated_descriptors.shape[0], query_id, dtype=np.int64))
    return DescriptorTable(stack_or_empty(chunks), concatenate_ids(ids))


"""
6. Spatial reranking with RANSAC over RootSIFT
"""


def load_image_features(
    metadata_row: pd.Series,
    feature_config: FeatureConfig,
    sift: cv2.SIFT | None,
) -> FeatureSet:
    return load_or_extract_rootsift(Path(metadata_row["img_path"]), sift, feature_config)


def match_descriptors_ratio(
    query_descs: FloatMatrix,
    candidate_descs: FloatMatrix,
    ratio: float,
) -> tuple[IntVector, IntVector]:
    if query_descs.shape[0] == 0 or candidate_descs.shape[0] < 2:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    sims = query_descs @ candidate_descs.T
    top2 = np.argpartition(-sims, kth=1, axis=1)[:, :2]
    rows = np.arange(sims.shape[0])[:, None]
    top2_sims = sims[rows, top2]
    order = np.argsort(-top2_sims, axis=1)
    sorted_idx = np.take_along_axis(top2, order, axis=1)
    sorted_sims = np.take_along_axis(top2_sims, order, axis=1)

    best_sim = np.clip(sorted_sims[:, 0], -1.0, 1.0)
    second_sim = np.clip(sorted_sims[:, 1], -1.0, 1.0)
    best_dist = np.sqrt(np.maximum(2.0 - 2.0 * best_sim, 0.0))
    second_dist = np.sqrt(np.maximum(2.0 - 2.0 * second_sim, 0.0))
    keep = best_dist < ratio * np.maximum(second_dist, 1e-6)

    q_idx = np.nonzero(keep)[0].astype(np.int64)
    c_idx = sorted_idx[keep, 0].astype(np.int64)
    return q_idx, c_idx


def count_ransac_inliers(
    query_features: FeatureSet,
    candidate_features: FeatureSet,
    sr_config: SpatialRerankConfig,
) -> int:
    q_descs = query_features.descriptors
    c_descs = candidate_features.descriptors

    if q_descs.shape[0] > sr_config.max_query_descriptors:
        order = np.argsort(-query_features.responses)[: sr_config.max_query_descriptors]
        q_descs = q_descs[order]
        q_pts = query_features.positions[order]
    else:
        q_pts = query_features.positions

    if c_descs.shape[0] > sr_config.max_candidate_descriptors:
        order = np.argsort(-candidate_features.responses)[: sr_config.max_candidate_descriptors]
        c_descs = c_descs[order]
        c_pts = candidate_features.positions[order]
    else:
        c_pts = candidate_features.positions

    q_idx, c_idx = match_descriptors_ratio(q_descs, c_descs, sr_config.ratio_test)
    if q_idx.shape[0] < sr_config.min_inliers:
        return 0

    src_pts = q_pts[q_idx].astype(np.float32)
    dst_pts = c_pts[c_idx].astype(np.float32)
    _, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC,
        sr_config.ransac_threshold,
        maxIters=sr_config.ransac_max_iters,
        confidence=sr_config.ransac_confidence,
    )
    if mask is None:
        return 0
    inliers = int(mask.sum())
    return inliers if inliers >= sr_config.min_inliers else 0


def spatial_rerank_topk(
    query_features: FeatureSet,
    initial_ranking: NDArray[np.int64],
    metadata: pd.DataFrame,
    feature_config: FeatureConfig,
    sr_config: SpatialRerankConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if not sr_config.enabled or query_features.descriptors.shape[0] == 0:
        return initial_ranking, np.zeros_like(initial_ranking)

    sift = create_sift(feature_config) if feature_config.detector == "sift" else None
    metadata_by_id = metadata.set_index("image_id", drop=False)

    top_n = min(sr_config.top_n, initial_ranking.shape[0])
    candidates = initial_ranking[:top_n]
    inliers = np.zeros(initial_ranking.shape[0], dtype=np.int64)

    for i, cand_id in enumerate(candidates.tolist()):
        cand_row = metadata_by_id.loc[cand_id]
        cand_features = load_image_features(cand_row, feature_config, sift)
        if cand_features.descriptors.shape[0] == 0:
            continue
        inliers[i] = count_ransac_inliers(query_features, cand_features, sr_config)

    verified_mask = inliers[:top_n] > 0
    verified_ids = candidates[verified_mask]
    verified_inliers = inliers[:top_n][verified_mask]
    order = np.argsort(-verified_inliers, kind="stable")
    verified_ids = verified_ids[order]

    not_verified_ids = candidates[~verified_mask]
    rest_ids = initial_ranking[top_n:]

    final_ranking = np.concatenate([verified_ids, not_verified_ids, rest_ids]).astype(np.int64)
    return final_ranking, inliers


"""
7. Query expansion: AQE and DQE in ASMK descriptor / codebook spaces
"""


def query_asmk(asmk: object, query_table: DescriptorTable) -> tuple[IntVector, NDArray[np.int64], FloatMatrix]:
    _metadata, query_ids, ranks, scores = asmk.query_ivf(
        query_table.descriptors,
        query_table.image_ids,
        progress=500,
    )
    return query_ids.astype(np.int64), ranks.astype(np.int64), scores.astype(np.float32)


def aggregated_descriptors_for_image(
    image_id: int,
    db_table: DescriptorTable,
) -> FloatMatrix:
    mask = db_table.image_ids == image_id
    return db_table.descriptors[mask]


def average_query_expansion(
    asmk: object,
    query_aggregated: FloatMatrix,
    verified_ids: NDArray[np.int64],
    db_table: DescriptorTable,
    n_positive: int,
) -> tuple[NDArray[np.int64], FloatMatrix]:
    expand_ids = verified_ids[:n_positive]
    chunks: list[FloatMatrix] = [query_aggregated]
    for image_id in expand_ids.tolist():
        chunks.append(aggregated_descriptors_for_image(int(image_id), db_table))

    expanded_descs = stack_or_empty(chunks)
    if expanded_descs.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)

    expanded_descs = l2_normalize_rows(expanded_descs)
    expanded_ids = np.zeros(expanded_descs.shape[0], dtype=np.int64)
    expanded_table = DescriptorTable(expanded_descs, expanded_ids)
    _qids, ranks, scores = query_asmk(asmk, expanded_table)
    return ranks[0].astype(np.int64), scores[0:1].astype(np.float32)


def get_codebook_centroids(asmk: object) -> FloatMatrix:
    """Extract codebook centroids from ASMK, trying common attribute paths."""
    codebook = getattr(asmk, "codebook", None)
    if codebook is None:
        raise AttributeError("ASMKMethod instance has no `codebook` attribute.")

    for attr in ("centroids", "centers", "C"):
        if hasattr(codebook, attr):
            return np.ascontiguousarray(getattr(codebook, attr), dtype=np.float32)

    if isinstance(codebook, dict):
        for key in ("centroids", "centers", "C"):
            if key in codebook:
                return np.ascontiguousarray(codebook[key], dtype=np.float32)

    raise AttributeError(
        f"Cannot locate centroids on codebook of type {type(codebook).__name__}. "
        f"Available attrs: {dir(codebook)}"
    )


def hard_assign_to_codebook_chunked(
    centroids: FloatMatrix,
    descriptors: FloatMatrix,
    chunk_size: int,
) -> IntVector:
    """Hard-assign descriptors to nearest centroid (cosine), chunked to avoid OOM."""
    if descriptors.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    out = np.empty(descriptors.shape[0], dtype=np.int64)
    centroids_t = centroids.T
    for start in range(0, descriptors.shape[0], chunk_size):
        end = min(start + chunk_size, descriptors.shape[0])
        sims = descriptors[start:end] @ centroids_t
        out[start:end] = np.argmax(sims, axis=1).astype(np.int64)
    return out


def build_db_word_assignments(
    asmk: object,
    db_table: DescriptorTable,
    chunk_size: int,
) -> IntVector:
    """Hard-assign every DB descriptor to a visual word once, chunked."""
    centroids = get_codebook_centroids(asmk)
    return hard_assign_to_codebook_chunked(centroids, db_table.descriptors, chunk_size)


def build_db_bow_matrix(
    db_word_assignments: IntVector,
    image_ids: IntVector,
    n_images: int,
    vocab_size: int,
) -> sparse.csr_matrix:
    """Build sparse term-frequency matrix [n_images, vocab_size] from precomputed word assignments."""
    data = np.ones(db_word_assignments.shape[0], dtype=np.float32)
    return sparse.csr_matrix(
        (data, (image_ids, db_word_assignments)),
        shape=(n_images, vocab_size),
        dtype=np.float32,
    )


def compute_idf_from_bow(bow_tf: sparse.csr_matrix) -> FloatVector:
    n_images = bow_tf.shape[0]
    doc_freq = np.asarray((bow_tf > 0).sum(axis=0)).ravel().astype(np.float32)
    return (np.log((n_images + 1) / (doc_freq + 1)) + 1.0).astype(np.float32)


def normalize_bow_tfidf(bow_tf: sparse.csr_matrix, idf: FloatVector) -> sparse.csr_matrix:
    weighted = bow_tf.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
    norms = np.maximum(norms, 1e-12).astype(np.float32)
    inv_norms = sparse.diags(1.0 / norms)
    return (inv_norms @ weighted).tocsr()


def query_to_bow_tfidf(
    query_words: IntVector,
    idf: FloatVector,
    vocab_size: int,
) -> FloatVector:
    tf = np.zeros(vocab_size, dtype=np.float32)
    if query_words.size == 0:
        return tf
    unique, counts = np.unique(query_words, return_counts=True)
    tf[unique] = counts.astype(np.float32)
    vec = tf * idf
    norm = np.linalg.norm(vec)
    return (vec / max(norm, 1e-12)).astype(np.float32)


def discriminative_query_expansion(
    query_aggregated: FloatMatrix,
    verified_ids: NDArray[np.int64],
    initial_ranking: NDArray[np.int64],
    initial_scores: FloatVector,
    bow_tfidf: sparse.csr_matrix,
    bow_idf: FloatVector,
    centroids: FloatMatrix,
    qe_config: QueryExpansionConfig,
    vocab_size: int,
) -> NDArray[np.int64]:
    pos_ids = verified_ids[: qe_config.n_positive]
    if pos_ids.shape[0] == 0:
        return initial_ranking

    not_pos_mask = ~np.isin(initial_ranking, pos_ids)
    candidate_pool = initial_ranking[not_pos_mask]
    candidate_scores = initial_scores[not_pos_mask]
    nonzero = candidate_scores > 0
    candidate_pool = candidate_pool[nonzero]
    candidate_scores = candidate_scores[nonzero]
    neg_order = np.argsort(candidate_scores, kind="stable")
    neg_ids = candidate_pool[neg_order[: qe_config.n_negative]]

    pos_vectors = bow_tfidf[pos_ids].toarray()
    neg_vectors = (
        bow_tfidf[neg_ids].toarray()
        if neg_ids.shape[0] > 0
        else np.empty((0, vocab_size), dtype=np.float32)
    )

    query_words = hard_assign_to_codebook_chunked(
        centroids, query_aggregated, qe_config.quantize_chunk_size,
    )
    query_vec = query_to_bow_tfidf(query_words, bow_idf, vocab_size)

    pos_words_set: set[int] = set()
    for vec in pos_vectors:
        pos_words_set.update(np.nonzero(vec)[0].tolist())
    pos_words_set.update(np.nonzero(query_vec)[0].tolist())
    pos_words = np.array(sorted(pos_words_set), dtype=np.int64)
    if pos_words.size == 0:
        return initial_ranking

    X_pos = pos_vectors[:, pos_words]
    X_neg = (
        neg_vectors[:, pos_words]
        if neg_vectors.shape[0] > 0
        else np.empty((0, pos_words.size), dtype=np.float32)
    )
    X_query = query_vec[pos_words].reshape(1, -1)
    X = np.vstack([X_pos, X_query, X_neg])
    y = np.concatenate([
        np.ones(X_pos.shape[0] + 1, dtype=np.float32),
        np.zeros(X_neg.shape[0], dtype=np.float32),
    ])

    if np.unique(y).size < 2:
        return initial_ranking

    svm = LinearSVC(C=qe_config.svm_c, max_iter=qe_config.svm_max_iter)
    svm.fit(X, y)
    weights = svm.coef_.ravel().astype(np.float32)

    db_truncated = bow_tfidf[:, pos_words]
    db_scores = np.asarray(db_truncated @ weights).ravel().astype(np.float32)
    return np.argsort(-db_scores, kind="stable").astype(np.int64)


"""
8. End-to-end retrieval and evaluation
"""


@dataclass(frozen=True)
class RetrievalOutput:
    method: str
    query_ids: IntVector
    rankings: NDArray[np.int64]
    scores: FloatMatrix


def run_retrieval_methods(
    asmk: object,
    metadata: pd.DataFrame,
    db_table: DescriptorTable,
    queries: list[GroundTruthQuery],
    query_bundles: list[QueryFeatureBundle],
    config: PipelineConfig,
) -> list[RetrievalOutput]:
    n_images = int(metadata["image_id"].max()) + 1
    vocab_size = config.asmk.codebook_size
    qe_config = config.query_expansion

    bow_tfidf: sparse.csr_matrix | None = None
    bow_idf: FloatVector | None = None
    centroids: FloatMatrix | None = None

    if qe_config.enable_dqe:
        print("Pre-computing DB BoW (one-time cost for DQE)")
        centroids = get_codebook_centroids(asmk)
        db_words = hard_assign_to_codebook_chunked(
            centroids, db_table.descriptors, qe_config.quantize_chunk_size,
        )
        bow_tf = build_db_bow_matrix(db_words, db_table.image_ids, n_images, vocab_size)
        bow_idf = compute_idf_from_bow(bow_tf)
        bow_tfidf = normalize_bow_tfidf(bow_tf, bow_idf)
        print(f"BoW tf-idf matrix: shape={bow_tfidf.shape}, nnz={bow_tfidf.nnz:,}")

    baseline_query_ids: list[int] = []
    baseline_ranks: list[NDArray[np.int64]] = []
    baseline_scores: list[FloatVector] = []
    sr_ranks: list[NDArray[np.int64]] = []
    aqe_ranks: list[NDArray[np.int64]] = []
    aqe_scores: list[FloatVector] = []
    dqe_ranks: list[NDArray[np.int64]] = []

    for query_id, (query, bundle) in enumerate(tqdm(
        list(zip(queries, query_bundles)), desc="Retrieve per query",
    )):
        if bundle.aggregated_descriptors.shape[0] == 0:
            continue

        single_table = DescriptorTable(
            bundle.aggregated_descriptors,
            np.zeros(bundle.aggregated_descriptors.shape[0], dtype=np.int64),
        )
        _qid, ranks, scores = query_asmk(asmk, single_table)
        baseline_query_ids.append(query_id)
        initial_ranking = ranks[0]
        initial_scores = scores[0]
        baseline_ranks.append(initial_ranking)
        baseline_scores.append(initial_scores)

        sr_ranking, _inliers = spatial_rerank_topk(
            bundle.raw_features, initial_ranking, metadata,
            config.features, config.spatial_rerank,
        )
        sr_ranks.append(sr_ranking)
        verified_ids = sr_ranking[: config.spatial_rerank.top_n]

        if qe_config.enable_aqe:
            aqe_ranking, aqe_score = average_query_expansion(
                asmk, bundle.aggregated_descriptors, verified_ids,
                db_table, qe_config.n_positive,
            )
            aqe_ranks.append(aqe_ranking)
            aqe_scores.append(aqe_score[0] if aqe_score.size > 0 else np.zeros(n_images, dtype=np.float32))

        if qe_config.enable_dqe and bow_tfidf is not None and bow_idf is not None and centroids is not None:
            dqe_ranking = discriminative_query_expansion(
                bundle.aggregated_descriptors, verified_ids,
                initial_ranking, initial_scores,
                bow_tfidf, bow_idf, centroids,
                qe_config, vocab_size,
            )
            dqe_ranks.append(dqe_ranking)

    query_ids_array = np.array(baseline_query_ids, dtype=np.int64)
    outputs = [
        RetrievalOutput("baseline_asmk", query_ids_array,
                        np.stack(baseline_ranks) if baseline_ranks else np.empty((0, n_images), dtype=np.int64),
                        np.stack(baseline_scores) if baseline_scores else np.empty((0, n_images), dtype=np.float32)),
        RetrievalOutput("asmk_sr", query_ids_array,
                        np.stack(sr_ranks) if sr_ranks else np.empty((0, n_images), dtype=np.int64),
                        np.stack(baseline_scores) if baseline_scores else np.empty((0, n_images), dtype=np.float32)),
    ]
    if qe_config.enable_aqe and aqe_ranks:
        outputs.append(RetrievalOutput(
            "asmk_sr_aqe", query_ids_array,
            np.stack(aqe_ranks),
            np.stack(aqe_scores),
        ))
    if qe_config.enable_dqe and dqe_ranks:
        outputs.append(RetrievalOutput(
            "asmk_sr_dqe", query_ids_array,
            np.stack(dqe_ranks),
            np.zeros((len(dqe_ranks), n_images), dtype=np.float32),
        ))
    return outputs


def average_precision_with_junk(
    ranked_ids: IntVector,
    positive_ids: set[int],
    junk_ids: set[int],
) -> float:
    if len(positive_ids) == 0:
        return float("nan")

    hits = 0
    precision_sum = 0.0
    seen: set[int] = set()
    effective_rank = 0
    for image_id in ranked_ids.tolist():
        if image_id in seen:
            continue
        seen.add(image_id)
        if image_id in junk_ids:
            continue
        effective_rank += 1
        if image_id in positive_ids:
            hits += 1
            precision_sum += hits / effective_rank

    return precision_sum / len(positive_ids)


def evaluate_method(
    method: str,
    queries: list[GroundTruthQuery],
    query_ids: IntVector,
    ranks: NDArray[np.int64],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_id, query_id in enumerate(query_ids.tolist()):
        query = queries[query_id]
        ap = average_precision_with_junk(ranks[row_id].astype(np.int64), query.positive_ids, query.junk_ids)
        rows.append({
            "method": method,
            "query_id": query_id,
            "dataset": query.dataset,
            "query_name": query.query_name,
            "landmark": query.landmark,
            "average_precision": ap,
        })
    ap_df = pd.DataFrame(rows)
    if ap_df.empty:
        return ap_df
    dataset_name = ap_df["dataset"].iloc[0]
    map_row = pd.DataFrame([{
        "method": method,
        "query_id": -1,
        "dataset": dataset_name,
        "query_name": "__mAP__",
        "landmark": "__mAP__",
        "average_precision": ap_df["average_precision"].mean(skipna=True),
    }])
    return pd.concat([ap_df, map_row], ignore_index=True)


"""
9. Output / reporting
"""


def export_retrieval_results(
    metadata: pd.DataFrame,
    queries: list[GroundTruthQuery],
    output: RetrievalOutput,
    output_dir: Path,
    topk: int,
) -> Path:
    metadata_by_id = metadata.set_index("image_id", drop=False)
    rows: list[dict[str, object]] = []

    for row_id, query_id in enumerate(output.query_ids.tolist()):
        query = queries[query_id]
        limit = min(topk, output.rankings.shape[1])
        for rank_position in range(limit):
            retrieved_id = int(output.rankings[row_id, rank_position])
            retrieved = metadata_by_id.loc[retrieved_id]
            rows.append({
                "method": output.method,
                "query_id": query_id,
                "query_name": query.query_name,
                "landmark": query.landmark,
                "rank": rank_position + 1,
                "retrieved_id": retrieved_id,
                "retrieved_basename": image_basename(retrieved["img_name"]),
                "retrieved_path": retrieved["img_path"],
                "score": float(output.scores[row_id, rank_position]) if output.scores.size > 0 else 0.0,
                "is_positive": retrieved_id in query.positive_ids,
                "is_junk": retrieved_id in query.junk_ids,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"retrieval_results_{output.method}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def export_metrics(metrics: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "map_metrics.csv"
    metrics.to_csv(output_path, index=False)
    return output_path


def print_map_summary(metrics: pd.DataFrame) -> None:
    summary = metrics[metrics["landmark"] == "__mAP__"]
    print("\nmAP summary")
    pivot = summary.pivot(index="method", columns="dataset", values="average_precision")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))


def run_pipeline(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    notify_event("Shi + ASMK retrieval pipeline started.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_or_build_metadata(config.dataset, rebuild=config.rebuild_metadata)
    tau_by_dataset = calibrate_taus(
        metadata,
        feature_config=config.features,
        burst_config=config.burst,
        calibration_config=config.calibration,
        output_dir=config.output_dir,
    )

    full_valid_metadata, _full_db_table, _full_query_table = build_descriptor_tables(
        metadata,
        config.features,
        config.burst,
        aggregate_queries=config.aggregate_queries,
        tau_by_dataset=tau_by_dataset,
        descriptor_workers=config.descriptor_workers,
        descriptor_chunksize=config.descriptor_chunksize,
    )
    full_valid_metadata.to_csv(config.output_dir / "valid_metadata.csv", index=False)

    dataset_names = sorted(full_valid_metadata["dataset"].unique())
    metrics_by_dataset: dict[str, pd.DataFrame] = {}

    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"  Running independent benchmark: {dataset_name}")
        print(f"{'='*60}")

        ds_metadata = full_valid_metadata[full_valid_metadata["dataset"] == dataset_name].copy()
        ds_metadata = ds_metadata.reset_index(drop=True)
        ds_metadata["image_id"] = ds_metadata.index.astype(np.int64)

        ds_output_dir = config.output_dir / dataset_name
        ds_output_dir.mkdir(parents=True, exist_ok=True)
        ds_metadata.to_csv(ds_output_dir / "valid_metadata.csv", index=False)

        ds_valid_metadata, ds_db, _ = build_descriptor_tables(
            ds_metadata,
            config.features,
            config.burst,
            aggregate_queries=config.aggregate_queries,
            tau_by_dataset=tau_by_dataset,
            descriptor_workers=config.descriptor_workers,
            descriptor_chunksize=config.descriptor_chunksize,
        )

        print(f"[{dataset_name}] Indexed images: {len(ds_valid_metadata)}")
        print(f"[{dataset_name}] Database descriptors after Shi aggregation: {ds_db.descriptors.shape[0]}")

        asmk = train_and_index_asmk(ds_db, config, dataset_name=dataset_name)

        gt_dir = gt_dir_for_dataset(config.ground_truth, dataset_name)
        if not gt_dir.exists():
            print(f"[{dataset_name}] No ground-truth directory found at {gt_dir}, skipping evaluation.")
            continue

        queries = load_ground_truth_for_dataset(ds_valid_metadata, config.ground_truth, dataset_name)
        query_bundles = build_ground_truth_query_bundles(
            ds_valid_metadata, queries, config.features, config.burst, tau_by_dataset[dataset_name],
        )

        outputs = run_retrieval_methods(
            asmk, ds_valid_metadata, ds_db, queries, query_bundles, config,
        )

        all_metrics_chunks: list[pd.DataFrame] = []
        for output in outputs:
            metrics = evaluate_method(output.method, queries, output.query_ids, output.rankings)
            all_metrics_chunks.append(metrics)
            export_retrieval_results(ds_valid_metadata, queries, output, ds_output_dir, topk=config.results_topk)

        all_metrics = pd.concat(all_metrics_chunks, ignore_index=True)
        metrics_path = export_metrics(all_metrics, ds_output_dir)
        print_map_summary(all_metrics)
        print(f"[{dataset_name}] Metrics saved to: {metrics_path}")
        metrics_by_dataset[dataset_name] = all_metrics

    notify_event("Shi + ASMK retrieval pipeline finished.")
    return metrics_by_dataset


if __name__ == "__main__":
    run_pipeline(PipelineConfig(
        descriptor_workers=32,
        descriptor_chunksize=4,
        asmk=ASMKConfig(
            gpu_id=0,
            codebook_size=65536,
            train_sample_size=2_600_000,
        ),
        burst=BurstConfig(tau=0.990),
    ))