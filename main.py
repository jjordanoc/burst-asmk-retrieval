"""
Shi et al. early burst detection + ASMK image retrieval pipeline.

The file is organized as notebook-ready sections. Run `run_pipeline(PipelineConfig())`
after the Oxford/Paris archives are available in the configured Colab paths.
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
    # strongest = np.argsort(-features.responses)[: config.max_features_per_image]
    # strongest.sort()
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
    descriptor_sigmoid_b: float = 0.78  # paper
    descriptor_sigmoid_w: float = 35.0  # paper
    use_scale_kernel: bool = True  # paper
    use_orientation_kernel: bool = True  # paper: Oxford/Paris
    scale_lambda: float = 1.0  # paper
    orientation_kappa: float = 4.0  # paper
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
    """
    Return Shi-style descriptors with shape [n_aggregated, 128].

    Features are graph nodes. Edges connect pairs whose affinity exceeds the
    threshold. Each connected component is averaged and L2-normalized.
    """
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
    codebook_size: int = 65_536  # paper
    gpu_id: int | None = None
    binary: bool = True  # paper: ASMK*
    use_idf: bool = True  # paper
    db_multiple_assignment: int = 1  # paper
    query_multiple_assignment: int = 5  # paper
    similarity_threshold: float = 0.0  # paper
    alpha: float = 3.0  # paper
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
) -> object:
    ASMKMethod = import_asmk_method()
    config.asmk.cache_dir.mkdir(parents=True, exist_ok=True)
    binary_label = "bin" if config.asmk.binary else "nobin"
    codebook_path = config.asmk.cache_dir / f"codebook_k{config.asmk.codebook_size}.pkl"
    ivf_path = config.asmk.cache_dir / (
        f"ivf_k{config.asmk.codebook_size}_{binary_label}_idf{int(config.asmk.use_idf)}.pkl"
    )

    train_descriptors = sample_training_descriptors(db_table.descriptors, config)
    asmk = ASMKMethod.initialize_untrained(asmk_params(config.asmk))
    asmk = asmk.train_codebook(train_descriptors, cache_path=str(codebook_path))
    ivf_builder = asmk.create_ivf_builder(cache_path=str(ivf_path))
    if not ivf_builder.loaded_from_cache:
        ivf_builder.add(db_table.descriptors, db_table.image_ids, progress=500)
    return asmk.add_ivf_builder(ivf_builder)


"""
5. Similarity search and retrieval evaluation
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


def load_ground_truth_queries(metadata: pd.DataFrame, config: GroundTruthConfig) -> list[GroundTruthQuery]:
    queries: list[GroundTruthQuery] = []
    for dataset_name in sorted(metadata["dataset"].unique()):
        gt_dir = gt_dir_for_dataset(config, str(dataset_name))
        if gt_dir.exists():
            queries.extend(load_ground_truth_for_dataset(metadata, config, str(dataset_name)))
    if not queries:
        raise RuntimeError("No Oxford/Paris ground-truth query files were found.")
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


def build_ground_truth_query_table(
    metadata: pd.DataFrame,
    queries: list[GroundTruthQuery],
    feature_config: FeatureConfig,
) -> DescriptorTable:
    metadata_by_basename = {
        (row.dataset, image_basename(row.img_name)): row.img_path
        for row in metadata.itertuples(index=False)
    }
    sift = create_sift(feature_config) if feature_config.detector == "sift" else None
    descriptor_chunks: list[FloatMatrix] = []
    query_id_chunks: list[IntVector] = []

    for query_id, query in enumerate(tqdm(queries, desc="Build GT query descriptors")):
        image_path = metadata_by_basename[(query.dataset, query.query_basename)]
        features = load_or_extract_rootsift(Path(image_path), sift, feature_config)
        query_features = filter_features_by_bbox(features, query.bbox)
        if query_features.descriptors.shape[0] == 0:
            print(f"[WARN] GT query has no features in bbox: {query.query_name}")
            continue
        descriptor_chunks.append(query_features.descriptors)
        query_id_chunks.append(np.full(query_features.descriptors.shape[0], query_id, dtype=np.int64))

    return DescriptorTable(stack_or_empty(descriptor_chunks), concatenate_ids(query_id_chunks))


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


def evaluate_ground_truth_map(
    metadata: pd.DataFrame,
    queries: list[GroundTruthQuery],
    query_ids: IntVector,
    ranks: NDArray[np.int64],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row_id, query_id in enumerate(query_ids.tolist()):
        query = queries[query_id]
        ap = average_precision_with_junk(
            ranks[row_id].astype(np.int64),
            query.positive_ids,
            query.junk_ids,
        )
        rows.append(
            {
                "query_id": query_id,
                "dataset": query.dataset,
                "query_name": query.query_name,
                "landmark": query.landmark,
                "average_precision": ap,
            }
        )

    ap_df = pd.DataFrame(rows)
    dataset_rows = [
        {
            "query_id": -1,
            "dataset": dataset,
            "query_name": "__mAP__",
            "landmark": "__mAP__",
            "average_precision": values["average_precision"].mean(skipna=True),
        }
        for dataset, values in ap_df.groupby("dataset")
    ]
    dataset_rows.append(
        {
            "query_id": -1,
            "dataset": "combined",
            "query_name": "__mAP__",
            "landmark": "__mAP__",
            "average_precision": ap_df["average_precision"].mean(skipna=True),
        }
    )
    return pd.concat([ap_df, pd.DataFrame(dataset_rows)], ignore_index=True)


def query_asmk(asmk: object, query_table: DescriptorTable) -> tuple[IntVector, NDArray[np.int64], FloatMatrix]:
    _metadata, query_ids, ranks, scores = asmk.query_ivf(
        query_table.descriptors,
        query_table.image_ids,
        progress=500,
    )
    return query_ids.astype(np.int64), ranks.astype(np.int64), scores.astype(np.float32)


"""
6. Retrieval output
"""


def export_retrieval_results(
    metadata: pd.DataFrame,
    queries: list[GroundTruthQuery],
    query_ids: IntVector,
    ranks: NDArray[np.int64],
    scores: FloatMatrix,
    output_dir: Path,
    topk: int,
) -> Path:
    metadata_by_id = metadata.set_index("image_id", drop=False)
    rows: list[dict[str, object]] = []

    for row_id, query_id in enumerate(query_ids.tolist()):
        query = queries[query_id]
        limit = min(topk, ranks.shape[1])
        for rank_position in range(limit):
            retrieved_id = int(ranks[row_id, rank_position])
            retrieved = metadata_by_id.loc[retrieved_id]
            rows.append(
                {
                    "query_id": query_id,
                    "query_name": query.query_name,
                    "landmark": query.landmark,
                    "rank": rank_position + 1,
                    "retrieved_id": retrieved_id,
                    "retrieved_basename": image_basename(retrieved["img_name"]),
                    "retrieved_path": retrieved["img_path"],
                    "score": float(scores[row_id, rank_position]),
                    "is_positive": retrieved_id in query.positive_ids,
                    "is_junk": retrieved_id in query.junk_ids,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retrieval_results.csv"
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
    for row in summary.itertuples(index=False):
        print(f"{row.dataset}: {row.average_precision:.4f}")


def run_pipeline(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    valid_metadata, db_table, _query_table = build_descriptor_tables(
        metadata,
        config.features,
        config.burst,
        aggregate_queries=config.aggregate_queries,
        tau_by_dataset=tau_by_dataset,
        descriptor_workers=config.descriptor_workers,
        descriptor_chunksize=config.descriptor_chunksize,
    )
    valid_metadata.to_csv(config.output_dir / "valid_metadata.csv", index=False)

    print(f"Indexed images: {len(valid_metadata)}")
    print(f"Database descriptors after Shi aggregation: {db_table.descriptors.shape[0]}")

    asmk = train_and_index_asmk(db_table, config)
    queries = load_ground_truth_queries(valid_metadata, config.ground_truth)
    gt_query_table = build_ground_truth_query_table(valid_metadata, queries, config.features)
    query_ids, ranks, scores = query_asmk(asmk, gt_query_table)
    metrics = evaluate_ground_truth_map(valid_metadata, queries, query_ids, ranks)

    metrics_path = export_metrics(metrics, config.output_dir)
    results_path = export_retrieval_results(
        valid_metadata,
        queries,
        query_ids,
        ranks,
        scores,
        config.output_dir,
        topk=config.results_topk,
    )

    print_map_summary(metrics)
    print(f"Metrics saved to: {metrics_path}")
    print(f"Retrieval results saved to: {results_path}")
    notify_event("Shi + ASMK retrieval pipeline finished.")
    return metrics, valid_metadata


if __name__ == "__main__":
    run_pipeline(PipelineConfig(
        descriptor_workers=32,
        descriptor_chunksize=4,
        asmk=ASMKConfig(
            gpu_id=0,
            # gpu_id=None,
            # codebook_size=65536,
            codebook_size=16384,
            # train_sample_size=2_600_000,
            train_sample_size=700_000,
        ),
        burst=BurstConfig(tau=0.990),
    ))