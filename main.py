"""
Shi et al. early burst detection + ASMK image retrieval pipeline.

The file is organized as notebook-ready sections. Run `run_pipeline(PipelineConfig())`
after the Oxford/Paris archives are available in the configured Colab paths.
"""

from __future__ import annotations

import math
import importlib
import os
import shutil
import sys
import tarfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
    max_features_per_image: int = 3000
    contrast_threshold: float = 0.01
    edge_threshold: float = 10.0
    cache_dir: Path = Path(f"{CONTENT_ROOT}/cache/features")


@dataclass(frozen=True)
class FeatureSet:
    descriptors: FloatMatrix
    scales: FloatVector
    orientations: FloatVector
    responses: FloatVector


def create_sift(config: FeatureConfig) -> cv2.SIFT:
    return cv2.SIFT_create(
        nfeatures=config.max_features_per_image,
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
        scales=np.empty((0,), dtype=np.float32),
        orientations=np.empty((0,), dtype=np.float32),
        responses=np.empty((0,), dtype=np.float32),
    )


def feature_cache_path(image_path: Path, config: FeatureConfig) -> Path:
    safe_name = str(image_path).strip("/").replace("/", "__")
    return config.cache_dir / f"{safe_name}.npz"


def extract_rootsift(image_path: Path, sift: cv2.SIFT) -> FeatureSet:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None or len(keypoints) == 0:
        return empty_feature_set()

    root_descriptors = rootsift(descriptors)
    scales = np.array([kp.size for kp in keypoints], dtype=np.float32)
    orientations = np.deg2rad(np.array([kp.angle if kp.angle >= 0 else 0.0 for kp in keypoints]))
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    return FeatureSet(root_descriptors, scales, orientations.astype(np.float32), responses)


def load_or_extract_rootsift(image_path: Path, sift: cv2.SIFT, config: FeatureConfig) -> FeatureSet:
    cache_path = feature_cache_path(image_path, config)
    if cache_path.exists():
        cached = np.load(cache_path)
        return FeatureSet(
            descriptors=cached["descriptors"].astype(np.float32),
            scales=cached["scales"].astype(np.float32),
            orientations=cached["orientations"].astype(np.float32),
            responses=cached["responses"].astype(np.float32),
        )

    features = extract_rootsift(image_path, sift)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        descriptors=features.descriptors,
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
    affinity_threshold: float = 0.78
    use_scale_kernel: bool = True
    use_orientation_kernel: bool = True
    scale_lambda: float = 1.0
    orientation_kappa: float = 4.0
    max_pairwise_features: int = 3000


def keep_strongest_features(features: FeatureSet, max_features: int) -> FeatureSet:
    if features.descriptors.shape[0] <= max_features:
        return features

    strongest = np.argsort(-features.responses)[:max_features]
    strongest.sort()
    return FeatureSet(
        descriptors=features.descriptors[strongest],
        scales=features.scales[strongest],
        orientations=features.orientations[strongest],
        responses=features.responses[strongest],
    )


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
    descriptors = l2_normalize_rows(features.descriptors)
    affinity = np.clip(descriptors @ descriptors.T, 0.0, 1.0).astype(np.float32)

    if config.use_scale_kernel:
        affinity *= scale_affinity(features.scales, config.scale_lambda)
    if config.use_orientation_kernel:
        affinity *= orientation_affinity(features.orientations, config.orientation_kappa)

    np.fill_diagonal(affinity, 1.0)
    return affinity


def aggregate_bursts(features: FeatureSet, config: BurstConfig) -> FloatMatrix:
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
    adjacency = sparse.csr_matrix(affinity >= config.affinity_threshold)
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
    codebook_size: int = 4096
    gpu_id: int | None = None
    binary: bool = False
    use_idf: bool = True
    db_multiple_assignment: int = 1
    query_multiple_assignment: int = 5
    similarity_threshold: float = 0.0
    alpha: float = 3.0
    topk: int | None = None
    train_sample_size: int = 250_000
    cache_dir: Path = Path(f"{CONTENT_ROOT}/cache/asmk")


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
    skip_reason: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    burst: BurstConfig = field(default_factory=BurstConfig)
    asmk: ASMKConfig = field(default_factory=ASMKConfig)
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
    args: tuple[ImageDescriptorTask, FeatureConfig, BurstConfig, bool],
) -> ImageDescriptorResult:
    task, feature_config, burst_config, aggregate_queries = args
    cv2.setNumThreads(1)
    sift = create_sift(feature_config)
    image_path = Path(task.img_path)

    features = load_or_extract_rootsift(image_path, sift, feature_config)
    if features.descriptors.shape[0] == 0:
        return ImageDescriptorResult(
            task=task,
            db_descriptors=empty_feature_set().descriptors,
            query_descriptors=empty_feature_set().descriptors,
            skip_reason="no SIFT features",
        )

    db_descriptors = aggregate_bursts(features, burst_config)
    query_descriptors = db_descriptors if aggregate_queries else features.descriptors
    if db_descriptors.shape[0] == 0 or query_descriptors.shape[0] == 0:
        return ImageDescriptorResult(
            task=task,
            db_descriptors=db_descriptors,
            query_descriptors=query_descriptors,
            skip_reason="empty descriptor aggregation",
        )

    return ImageDescriptorResult(
        task=task,
        db_descriptors=db_descriptors,
        query_descriptors=query_descriptors.astype(np.float32),
    )


def iter_descriptor_results(
    tasks: list[ImageDescriptorTask],
    feature_config: FeatureConfig,
    burst_config: BurstConfig,
    aggregate_queries: bool,
    max_workers: int,
    chunksize: int,
) -> Iterator[ImageDescriptorResult]:
    if chunksize < 1:
        raise ValueError(f"descriptor_chunksize must be >= 1, got {chunksize}")

    worker_args = (
        (task, feature_config, burst_config, aggregate_queries)
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
            }
        )
        db_descriptor_chunks.append(result.db_descriptors)
        db_id_chunks.append(np.full(result.db_descriptors.shape[0], image_id, dtype=np.int64))
        query_descriptor_chunks.append(result.query_descriptors)
        query_id_chunks.append(np.full(result.query_descriptors.shape[0], image_id, dtype=np.int64))

    valid_metadata = pd.DataFrame(valid_records)
    if valid_metadata.empty:
        raise RuntimeError("No valid images remained after feature extraction.")

    db_table = DescriptorTable(stack_or_empty(db_descriptor_chunks), concatenate_ids(db_id_chunks))
    query_table = DescriptorTable(stack_or_empty(query_descriptor_chunks), concatenate_ids(query_id_chunks))
    return valid_metadata, db_table, query_table


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
    codebook_path = config.asmk.cache_dir / "codebook.pkl"
    ivf_path = config.asmk.cache_dir / "ivf.pkl"

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


def average_precision(ranked_ids: IntVector, relevant_ids: set[int]) -> float:
    if len(relevant_ids) == 0:
        return float("nan")

    hits = 0
    precision_sum = 0.0
    seen: set[int] = set()
    for rank, image_id in enumerate(ranked_ids.tolist(), start=1):
        if image_id in seen:
            continue
        seen.add(image_id)
        if image_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant_ids)


def evaluate_map(
    metadata: pd.DataFrame,
    query_ids: IntVector,
    ranks: NDArray[np.int64],
) -> pd.DataFrame:
    metadata_by_id = metadata.set_index("image_id", drop=False)
    rows: list[dict[str, object]] = []

    for row_id, query_id in enumerate(query_ids.tolist()):
        query = metadata_by_id.loc[query_id]
        relevant = metadata[
            (metadata["dataset"] == query["dataset"])
            & (metadata["class_name"] == query["class_name"])
            & (metadata["image_id"] != query_id)
        ]["image_id"]
        ranked_ids = ranks[row_id].astype(np.int64)
        ranked_ids = ranked_ids[ranked_ids != query_id]
        ap = average_precision(ranked_ids, set(relevant.astype(int).tolist()))
        rows.append(
            {
                "query_id": query_id,
                "dataset": query["dataset"],
                "class_name": query["class_name"],
                "average_precision": ap,
            }
        )

    ap_df = pd.DataFrame(rows)
    dataset_rows = [
        {
            "query_id": -1,
            "dataset": dataset,
            "class_name": "__mAP__",
            "average_precision": values["average_precision"].mean(skipna=True),
        }
        for dataset, values in ap_df.groupby("dataset")
    ]
    dataset_rows.append(
        {
            "query_id": -1,
            "dataset": "combined",
            "class_name": "__mAP__",
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
    query_ids: IntVector,
    ranks: NDArray[np.int64],
    scores: FloatMatrix,
    output_dir: Path,
    topk: int,
) -> Path:
    metadata_by_id = metadata.set_index("image_id", drop=False)
    rows: list[dict[str, object]] = []

    for row_id, query_id in enumerate(query_ids.tolist()):
        query = metadata_by_id.loc[query_id]
        limit = min(topk, ranks.shape[1])
        for rank_position in range(limit):
            retrieved_id = int(ranks[row_id, rank_position])
            retrieved = metadata_by_id.loc[retrieved_id]
            rows.append(
                {
                    "query_id": query_id,
                    "query_path": query["img_path"],
                    "rank": rank_position + 1,
                    "retrieved_id": retrieved_id,
                    "retrieved_path": retrieved["img_path"],
                    "score": float(scores[row_id, rank_position]),
                    "same_dataset": query["dataset"] == retrieved["dataset"],
                    "same_class": query["class_name"] == retrieved["class_name"],
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
    summary = metrics[metrics["class_name"] == "__mAP__"]
    print("\nmAP summary")
    for row in summary.itertuples(index=False):
        print(f"{row.dataset}: {row.average_precision:.4f}")


def run_pipeline(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    notify_event("Shi + ASMK retrieval pipeline started.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_or_build_metadata(config.dataset, rebuild=config.rebuild_metadata)
    valid_metadata, db_table, query_table = build_descriptor_tables(
        metadata,
        config.features,
        config.burst,
        aggregate_queries=config.aggregate_queries,
        descriptor_workers=config.descriptor_workers,
        descriptor_chunksize=config.descriptor_chunksize,
    )
    valid_metadata.to_csv(config.output_dir / "valid_metadata.csv", index=False)

    print(f"Indexed images: {len(valid_metadata)}")
    print(f"Database descriptors after Shi aggregation: {db_table.descriptors.shape[0]}")
    print(f"Query descriptors: {query_table.descriptors.shape[0]}")

    asmk = train_and_index_asmk(db_table, config)
    query_ids, ranks, scores = query_asmk(asmk, query_table)
    metrics = evaluate_map(valid_metadata, query_ids, ranks)

    metrics_path = export_metrics(metrics, config.output_dir)
    results_path = export_retrieval_results(
        valid_metadata,
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
        descriptor_workers=16,
        descriptor_chunksize=4,
        burst=BurstConfig(
            max_pairwise_features=1500,
        ),
        features=FeatureConfig(
            max_features_per_image=1500,
        )
    ))