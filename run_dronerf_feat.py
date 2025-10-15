### script version of Generate DroneRF Features Notebook

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
from sklearn.model_selection import train_test_split
from spafe.features.lfcc import lfcc
import spafe.utils.vis as vis
from scipy.signal import get_window
import scipy.fftpack as fft
from scipy import signal
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm

from loading_functions import *
from file_paths import *
from feat_gen_functions import *

import importlib

# Dataset Info
main_folder = dronerf_raw_path
t_seg = 20
fs = 40e6 #40 MHz
samples_per_segment = int(t_seg/1e3*fs)

# Streaming configuration for chunk-wise feature extraction
chunk_size_samples = samples_per_segment * 4  # four segments per read (~memory/perf tradeoff)
segments_per_batch = 64  # how many segments to accumulate before flushing to disk
stream_dtype = np.float32
max_workers_default: Optional[int] = None  # None lets ThreadPoolExecutor decide

# Resume helpers for notebook environments
resume_existing_progress = True  # set False to recompute from scratch
reset_existing_progress = False  # set True to ignore any detected progress

n_per_seg = 1024 # length of each segment (powers of 2)
n_overlap_spec = 120
win_type = 'hamming' # make ends of each segment match
high_low = 'L' #'L', 'H' # high or low range of frequency
feature_to_save = ['PSD'] # what features to generate and save: SPEC or PSD
format_to_save = ['IMG'] # IMG or ARR or RAW
to_add = True
spec_han_window = np.hanning(n_per_seg)

# Image properties
dim_px = (224, 224) # dimension of image pixels
dpi = 100

# Raw input len
v_samp_len = 10000

# data saving folders
features_folder = dronerf_feat_path
date_string = date.today()
# folder naming: ARR_FEAT_NFFT_SAMPLELENGTH
arr_spec_folder = "ARR_SPEC_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
arr_psd_folder = "ARR_PSD_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
img_spec_folder = "IMG_SPEC_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
img_psd_folder = "IMG_PSD_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
raw_folder = 'RAW_VOLT_'+str(v_samp_len)+"_"+str(t_seg)+"/" # high and low frequency stacked together

LOG_INTERVAL = 10
SLOW_STEP_THRESHOLD = 2.0


@dataclass
class ExtractionConfig:
    fs: float
    n_per_seg: int
    n_overlap_spec: int
    i_hl: int
    v_samp_len: int
    sa_save: bool
    si_save: bool
    pa_save: bool
    pi_save: bool
    raw_save: bool
    win_type: str
    spec_window: np.ndarray
    spec_interp_shape: Tuple[int, int]


@dataclass
class SegmentResult:
    bi_label: Any
    drone_label: Any
    model_label: Any
    psd: Optional[np.ndarray]
    spec_image: Optional[np.ndarray]
    spec_array: Optional[np.ndarray]
    raw: Optional[np.ndarray]
    durations: Dict[str, float]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DroneRF features with optional parallelism.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max_workers_default,
        help="Number of worker threads to use (default: auto)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=LOG_INTERVAL,
        help="How many processed segments to summarize in progress logs.",
    )
    parser.add_argument(
        "--slow-step-threshold",
        type=float,
        default=SLOW_STEP_THRESHOLD,
        help="Emit a warning when an individual step exceeds this duration in seconds.",
    )
    return parser.parse_args(argv)


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def _compute_features(record: Dict[str, Any], cfg: ExtractionConfig) -> SegmentResult:
    durations = {"raw": 0.0, "psd": 0.0, "spec": 0.0}
    segment = record['segment']
    d_real = segment[cfg.i_hl]

    raw = None
    if cfg.raw_save:
        start = time.perf_counter()
        t = np.arange(0, segment.shape[1])
        f_high = interpolate.interp1d(t, segment[0])
        f_low = interpolate.interp1d(t, segment[1])
        tt = np.linspace(0, segment.shape[1] - 1, num=cfg.v_samp_len)
        raw = np.stack((f_high(tt), f_low(tt)), axis=0).astype(np.float32, copy=False)
        durations["raw"] = time.perf_counter() - start

    psd = None
    if cfg.pa_save or cfg.pi_save:
        start = time.perf_counter()
        _, psd_vals = signal.welch(d_real, cfg.fs, window=cfg.win_type, nperseg=cfg.n_per_seg)
        psd = psd_vals.astype(np.float32, copy=False)
        durations["psd"] = time.perf_counter() - start

    spec_image = None
    spec_array = None
    if cfg.sa_save or cfg.si_save:
        start = time.perf_counter()
        _, _, Sxx = signal.spectrogram(
            d_real,
            fs=cfg.fs,
            window=cfg.spec_window,
            nperseg=cfg.n_per_seg,
            noverlap=cfg.n_overlap_spec,
            scaling='density',
            mode='psd',
        )
        spec_image = Sxx.astype(np.float32, copy=False)
        if cfg.sa_save:
            spec_array = interpolate_2d(spec_image, cfg.spec_interp_shape).astype(np.float32, copy=False)
        durations["spec"] = time.perf_counter() - start

    return SegmentResult(
        bi_label=record['bi_label'],
        drone_label=record['four_label'],
        model_label=record['ten_label'],
        psd=psd,
        spec_image=spec_image,
        spec_array=spec_array,
        raw=raw,
        durations=durations,
    )


def infer_saved_array_progress(folder_path: str, descriptor: str) -> Optional[Dict[str, Any]]:
    if not os.path.isdir(folder_path):
        return None

    records = []
    for name in os.listdir(folder_path):
        if not name.lower().endswith('.npy'):
            continue
        try:
            batch_idx = int(name.rsplit('_', 1)[-1].split('.')[0])
        except ValueError:
            continue

        file_path = os.path.join(folder_path, name)
        try:
            payload = np.load(file_path, allow_pickle=True).item()
            feat = payload.get('feat')
            seg_count = len(feat) if feat is not None else 0
        except Exception as exc:
            print(f"Warning: unable to inspect '{file_path}' while inferring progress ({exc})")
            continue

        records.append((batch_idx, seg_count))

    if not records:
        return None

    records.sort()
    processed = sum(seg_count for _, seg_count in records)
    next_batch = records[-1][0] + 1
    return {"source": descriptor, "processed_segments": processed, "next_batch_index": next_batch}


def infer_saved_image_progress(folder_path: str, descriptor: str) -> Optional[Dict[str, Any]]:
    if not os.path.isdir(folder_path):
        return None

    unique_positions = set()
    for name in os.listdir(folder_path):
        if not name.lower().endswith('.jpg'):
            continue
        parts = name[:-4].split('_')
        if len(parts) < 3:
            continue
        try:
            group_counter = int(parts[-2])
            count_in_group = int(parts[-1])
        except ValueError:
            continue
        unique_positions.add((group_counter, count_in_group))

    if not unique_positions:
        return None

    processed = len(unique_positions)
    next_batch = max(group for group, _ in unique_positions) + 1
    return {"source": descriptor, "processed_segments": processed, "next_batch_index": next_batch}


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    os.makedirs(features_folder, exist_ok=True)
    existing_folders = os.listdir(features_folder)

    if high_low == 'H':
        i_hl = 0
    elif high_low == 'L':
        i_hl = 1
    else:
        raise ValueError(f"Unsupported high_low value: {high_low}")

    sa_save = False   # spec array
    si_save = False   # spec image
    pa_save = False   # psd array
    pi_save = False   # psd image
    raw_save = False  # raw high low signals

    if 'SPEC' in feature_to_save:
        if 'ARR' in format_to_save:
            if arr_spec_folder not in existing_folders or to_add:
                try:
                    os.mkdir(features_folder+arr_spec_folder)
                except Exception:
                    print('folder already exist - adding')
                sa_save = True
                print('Generating SPEC in ARRAY format')
            else:
                print('Spec Arr folder already exists')
        if 'IMG' in format_to_save:
            if img_spec_folder not in existing_folders or to_add:
                try:
                    os.mkdir(features_folder+img_spec_folder)
                except Exception:
                    print('folder already exist - adding')
                si_save = True
                print('Generating SPEC in IMAGE format')
            else:
                print('Spec Arr folder already exists')
    if 'PSD' in feature_to_save:
        if 'ARR' in format_to_save:
            if arr_psd_folder not in existing_folders or to_add:
                try:
                    os.mkdir(features_folder+arr_psd_folder)
                except Exception:
                    print('folder already exist - adding')
                pa_save = True
                print('Generating PSD in ARRAY format')
            else:
                print('PSD Arr folder already exists')
        if 'IMG' in format_to_save:
            if img_psd_folder not in existing_folders or to_add:
                try:
                    os.mkdir(features_folder+img_psd_folder)
                except Exception:
                    print('folder already exist - adding')
                pi_save = True
                print('Generating PSD in IMAGE format')
            else:
                print('PSD Arr folder already exists')

    if 'RAW' in feature_to_save:
        if raw_folder in existing_folders or to_add:
            try:
                os.mkdir(features_folder+raw_folder)
            except Exception:
                print('RAW V folder already exists')
            raw_save = True

    if all([not sa_save, not si_save, not pa_save, not pi_save, not raw_save]):
        print('Features Already Exist - Do Not Generate')
        return

    print(
        f'Streaming DroneRF segments with chunk_size_samples={chunk_size_samples} '
        f'and segments_per_batch={segments_per_batch}'
    )

    max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else max_workers_default
    if max_workers in (None, 0):
        max_workers = os.cpu_count() or 1
    print(f"Using {max_workers} worker process(es) for feature extraction")

    cfg = ExtractionConfig(
        fs=fs,
        n_per_seg=n_per_seg,
        n_overlap_spec=n_overlap_spec,
        i_hl=i_hl,
        v_samp_len=v_samp_len,
        sa_save=sa_save,
        si_save=si_save,
        pa_save=pa_save,
        pi_save=pi_save,
        raw_save=raw_save,
        win_type=win_type,
        spec_window=spec_han_window,
        spec_interp_shape=dim_px,
    )

    def gather_resume_state() -> Optional[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        if sa_save:
            info = infer_saved_array_progress(features_folder + arr_spec_folder, "SPEC arrays")
            if info:
                candidates.append(info)
        if pa_save:
            info = infer_saved_array_progress(features_folder + arr_psd_folder, "PSD arrays")
            if info:
                candidates.append(info)
        if raw_save:
            info = infer_saved_array_progress(features_folder + raw_folder, "RAW arrays")
            if info:
                candidates.append(info)
        if si_save:
            info = infer_saved_image_progress(os.path.join(features_folder, img_spec_folder), "SPEC images")
            if info:
                candidates.append(info)
        if pi_save:
            info = infer_saved_image_progress(os.path.join(features_folder, img_psd_folder), "PSD images")
            if info:
                candidates.append(info)

        if not candidates:
            return None

        max_processed = max(info["processed_segments"] for info in candidates)
        top_candidates = [info for info in candidates if info["processed_segments"] == max_processed]
        resume_from = top_candidates[0]

        if len(top_candidates) > 1 or len({info["processed_segments"] for info in candidates}) > 1:
            sources = ", ".join(f"{info['source']} ({info['processed_segments']})" for info in candidates)
            print(
                "Detected differing progress markers across outputs; using the largest count. "
                f"Details: {sources}"
            )

        return resume_from

    def flush_batch(
        batch_idx: int,
        BILABEL: List[Any],
        DRONELABEL: List[Any],
        MODELALBEL: List[Any],
        F_SPEC: List[np.ndarray],
        F_PSD: List[np.ndarray],
        F_V: List[np.ndarray],
    ) -> Tuple[List[Any], List[Any], List[Any], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        if not BILABEL:
            return BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V

        if sa_save and F_SPEC:
            save_array_rf(
                features_folder + arr_spec_folder,
                F_SPEC,
                BILABEL,
                DRONELABEL,
                MODELALBEL,
                'SPEC',
                n_per_seg,
                batch_idx,
            )

        if pa_save and F_PSD:
            save_array_rf(
                features_folder + arr_psd_folder,
                F_PSD,
                BILABEL,
                DRONELABEL,
                MODELALBEL,
                'PSD',
                n_per_seg,
                batch_idx,
            )

        if raw_save and F_V:
            save_array_rf(
                features_folder + raw_folder,
                F_V,
                BILABEL,
                DRONELABEL,
                MODELALBEL,
                'RAW',
                '',
                batch_idx,
            )

        return [], [], [], [], [], []

    BILABEL: List[Any] = []
    DRONELABEL: List[Any] = []
    MODELALBEL: List[Any] = []
    F_PSD: List[np.ndarray] = []
    F_SPEC: List[np.ndarray] = []
    F_V: List[np.ndarray] = []

    resume_state = None
    processed_segments = 0

    if reset_existing_progress:
        print("Reset flag enabled: ignoring any existing outputs. Be careful of name collisions.")
    elif resume_existing_progress:
        resume_state = gather_resume_state()
        if resume_state:
            processed_segments = resume_state["processed_segments"]
            print(
                "Resuming from previously saved progress provided by "
                f"{resume_state['source']}: skipping {processed_segments} segments"
            )
        else:
            print("No existing progress detected; starting from the first segment.")

    batch_index = resume_state["next_batch_index"] if resume_state else 0
    count_in_batch = 0

    segment_iterator = iter_dronerf_segments(
        main_folder,
        t_seg,
        chunk_size_samples=chunk_size_samples,
        dtype=stream_dtype,
        skip_segments=processed_segments,
    )

    total_processed = processed_segments
    progress = tqdm(desc='Generating DroneRF features', unit='seg', initial=processed_segments)
    segments_since_log = 0
    rolling_durations = {"raw": 0.0, "psd": 0.0, "spec": 0.0, "save": 0.0}
    rolling_flush_duration = 0.0
    rolling_flush_events = 0

    def log_summary(force: bool = False) -> None:
        nonlocal segments_since_log, rolling_durations, rolling_flush_duration, rolling_flush_events
        if not force and segments_since_log < args.log_interval:
            return
        if segments_since_log == 0:
            return
        avg_raw = rolling_durations["raw"] / segments_since_log if segments_since_log else 0.0
        avg_psd = rolling_durations["psd"] / segments_since_log if segments_since_log else 0.0
        avg_spec = rolling_durations["spec"] / segments_since_log if segments_since_log else 0.0
        avg_save = rolling_durations["save"] / segments_since_log if segments_since_log else 0.0
        avg_flush = (
            rolling_flush_duration / rolling_flush_events if rolling_flush_events else 0.0
        )
        progress.write(
            f"Summary ({segments_since_log} segs ending at {total_processed}): "
            f"raw={avg_raw:.2f}s psd={avg_psd:.2f}s spec={avg_spec:.2f}s "
            f"save={avg_save:.2f}s flush={avg_flush:.2f}s"
        )
        segments_since_log = 0
        rolling_durations = {"raw": 0.0, "psd": 0.0, "spec": 0.0, "save": 0.0}
        rolling_flush_duration = 0.0
        rolling_flush_events = 0

    def process_segment(result: SegmentResult, batch_pos: int) -> None:
        nonlocal count_in_batch, total_processed, segments_since_log, batch_index
        nonlocal rolling_flush_duration, rolling_flush_events
        nonlocal BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V

        dur_raw = result.durations.get("raw", 0.0)
        dur_psd = result.durations.get("psd", 0.0)
        dur_spec = result.durations.get("spec", 0.0)
        dur_save = 0.0

        if result.raw is not None:
            F_V.append(result.raw)

        if result.psd is not None and pa_save:
            F_PSD.append(result.psd)

        if result.psd is not None and pi_save:
            save_start = time.perf_counter()
            psd_fig = plot_feat(result.psd, dim_px, dpi, to_show=False, show_axis=False)
            psd_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            save_psd_image_rf(
                features_folder,
                img_psd_folder,
                result.model_label,
                batch_index,
                batch_pos,
                psd_fig,
                dim_px,
                dpi,
            )
            dur_save += time.perf_counter() - save_start

        if result.spec_array is not None and sa_save:
            F_SPEC.append(result.spec_array)

        if result.spec_image is not None and si_save:
            save_start = time.perf_counter()
            spec_fig = plot_feat(result.spec_image, dim_px, dpi, to_show=False, show_axis=False)
            spec_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            for ax in spec_fig.axes:
                ax.axis('off')
            save_spec_image_fig_rf(
                features_folder,
                img_spec_folder,
                result.model_label,
                batch_index,
                batch_pos,
                spec_fig,
                dpi,
            )
            dur_save += time.perf_counter() - save_start

        BILABEL.append(result.bi_label)
        DRONELABEL.append(result.drone_label)
        MODELALBEL.append(result.model_label)

        count_in_batch += 1
        total_processed += 1
        progress.update(1)
        segments_since_log += 1

        rolling_durations["raw"] += dur_raw
        rolling_durations["psd"] += dur_psd
        rolling_durations["spec"] += dur_spec
        rolling_durations["save"] += dur_save

        if any(duration > args.slow_step_threshold for duration in (dur_raw, dur_psd, dur_spec, dur_save)):
            progress.write(
                f"Slow segment {total_processed} (batch {batch_index}, idx {batch_pos}): "
                f"raw={dur_raw:.2f}s psd={dur_psd:.2f}s spec={dur_spec:.2f}s save={dur_save:.2f}s"
            )

        log_summary()

        if count_in_batch >= segments_per_batch:
            segments_in_flush = count_in_batch
            flush_start = time.perf_counter()
            flush_batch_result = flush_batch(
                batch_index, BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V
            )
            flush_duration = time.perf_counter() - flush_start
            rolling_flush_duration += flush_duration
            rolling_flush_events += 1
            progress.write(
                f"Flushed batch {batch_index} with {segments_in_flush} segments in {flush_duration:.2f}s"
            )
            if flush_duration > args.slow_step_threshold:
                progress.write(
                    f"Slow flush for batch {batch_index}: duration {flush_duration:.2f}s"
                )
            BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V = flush_batch_result
            batch_idx_next = batch_index + 1
            log_summary(force=True)
            batch_index = batch_idx_next
            count_in_batch = 0

    compute_features = partial(_compute_features, cfg=cfg)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch in batched(segment_iterator, segments_per_batch):
            batch_start = count_in_batch
            for idx, result in enumerate(executor.map(compute_features, batch)):
                process_segment(result, batch_start + idx)

    if count_in_batch or BILABEL:
        segments_in_flush = count_in_batch
        flush_start = time.perf_counter()
        BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V = flush_batch(
            batch_index, BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V
        )
        flush_duration = time.perf_counter() - flush_start
        rolling_flush_duration += flush_duration
        rolling_flush_events += 1
        progress.write(
            f"Flushed final batch {batch_index} with {segments_in_flush} segments in {flush_duration:.2f}s"
        )
        if flush_duration > args.slow_step_threshold:
            progress.write(
                f"Slow flush for final batch {batch_index}: duration {flush_duration:.2f}s"
            )

    log_summary(force=True)
    progress.close()
    print(f"Completed feature extraction for {total_processed} segments.")


if __name__ == "__main__":
    main()
