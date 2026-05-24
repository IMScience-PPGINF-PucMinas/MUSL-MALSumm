import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import listdir
from os.path import join
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from scipy.stats import kendalltau, spearmanr

from evaluation.evaluation_metrics import evaluate_summary
from inference.generate_summary import generate_summary
from model.layers.summarizer import xLSTM
from utils.utils import get_paths, setup_logging

import argparse

setup_logging()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(
    'Using device: ' + str(DEVICE)
    + (f' ({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else '')
)


def load_video_data(
    dataset: str,
    data_path: str,
    video: str,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, int, np.ndarray, Optional[str]]:
    dataset_lower = dataset.lower()

    with h5py.File(data_path, 'r') as hdf:
        frame_features = torch.from_numpy(
            np.array(hdf[f'{video}/features'], dtype=np.float32)
        ).view(-1, 1024)
        sb = np.array(hdf[f'{video}/change_points'])

        if dataset_lower in ('summe', 'tvsum'):
            user_summary = np.array(hdf[f'{video}/user_summary'])
            n_frames = int(np.array(hdf[f'{video}/n_frames']))
            positions = np.array(hdf[f'{video}/picks'], dtype=np.int64)
            video_name = (
                str(np.array(hdf[f'{video}/video_name']).astype(str))
                if f'{video}/video_name' in hdf
                else None
            )
        elif dataset_lower == 'mrhisum':
            user_summary = np.array(hdf[f'{video}/gt_summary'])
            n_frames = frame_features.shape[0]
            positions = np.arange(n_frames, dtype=np.int64)
            video_name = None
        else:
            raise ValueError(f'Unsupported dataset: {dataset!r}')

    return frame_features, user_summary, sb, n_frames, positions, video_name


def _find_epoch_files(model_path: str) -> List[str]:
    files = [f for f in listdir(model_path) if re.match(r'epoch-\d+\.pkl', f)]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))


def _load_best_epoch_from_fscores(model_path: str) -> Optional[int]:
    fscores_path = join(model_path, 'f_scores.txt')
    if not os.path.exists(fscores_path):
        return None
    with open(fscores_path) as fp:
        content = fp.read().strip()
    try:
        scores = json.loads(content)
    except json.JSONDecodeError:
        scores = [float(x) for x in content.splitlines()]
    return int(np.argmax(scores))


def _load_model(model_path: str, fname: str, model_kwargs: dict) -> torch.nn.Module:
    model = xLSTM(**model_kwargs)
    state = torch.load(join(model_path, fname), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def run_inference(
    model: torch.nn.Module,
    data_path: str,
    keys: List[str],
    eval_method: str,
    save_summary: bool,
    dataset: str,
    verbose: bool = False,
) -> tuple:
    dataset_lower = dataset.lower()
    is_summe = dataset_lower == 'summe'

    video_fscores: List[float] = []
    video_kendalls: List[float] = []
    video_spearmans: List[float] = []
    video_summaries: Dict = {}
    video_names: Dict = {}

    for video in keys:
        if is_summe:
            try:
                if int(video.split('_')[1]) > 25:
                    continue
            except (IndexError, ValueError):
                pass

        frame_features, user_summary, sb, n_frames, positions, vname = load_video_data(
            dataset, data_path, video
        )

        with torch.no_grad():
            scores, _ = model(frame_features.to(DEVICE))
            scores = scores.squeeze(0).cpu().numpy().tolist()

        summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
        f_score = evaluate_summary(summary, user_summary, eval_method)

        frame_init_scores = np.asarray(scores, dtype=np.float64)
        frame_scores = np.zeros(n_frames, dtype=np.float64)
        pos = positions.astype(np.int64)
        if pos[-1] != n_frames:
            pos = np.concatenate([pos, [n_frames]])
        for i in range(len(pos) - 1):
            frame_scores[pos[i]:pos[i + 1]] = (
                frame_init_scores[i] if i < len(frame_init_scores) else 0.0
            )

        gt_importance = user_summary.mean(axis=0) if user_summary.ndim > 1 else user_summary

        if frame_scores.shape[0] != gt_importance.shape[0]:
            logging.warning(
                f'Shape mismatch for {video}: '
                f'pred={frame_scores.shape[0]}, gt={gt_importance.shape[0]} — skipping correlations'
            )
            ktau = spr = float('nan')
        else:
            ktau, _ = kendalltau(frame_scores, gt_importance)
            spr, _ = spearmanr(frame_scores, gt_importance)

        video_fscores.append(f_score)
        video_kendalls.append(ktau)
        video_spearmans.append(spr)
        video_summaries[video] = summary
        if is_summe:
            video_names[video] = vname

        if verbose:
            logging.info(f'  {video} ({vname}): F1={f_score:.2f}%  τ={ktau:.4f}  ρ={spr:.4f}')

        if save_summary:
            out = {str(i): int(v) for i, v in enumerate(summary)}
            fname = f'{video}_summary.json'
            with open(fname, 'w') as fp:
                json.dump(out, fp, indent=4)
            print(f'Summary saved → {fname}')

    mean_fscore = float(np.nanmean(video_fscores))
    mean_kendall = float(np.nanmean(video_kendalls))
    mean_spearman = float(np.nanmean(video_spearmans))

    if is_summe:
        return mean_fscore, mean_kendall, mean_spearman, video_summaries, video_names
    return mean_fscore, mean_kendall, mean_spearman, video_summaries


def _find_fold_dirs(split_path: str) -> List[str]:
    """Return sorted list of fold sub-directories inside a split path."""
    if not os.path.isdir(split_path):
        return []
    dirs = [d for d in os.listdir(split_path)
            if re.match(r'fold\d+', d) and os.path.isdir(join(split_path, d))]
    return sorted(dirs, key=lambda d: int(re.findall(r'\d+', d)[0]))


def _scan_split_worker(args: tuple) -> Tuple[int, int, Dict]:
    (split_id, split_path, dataset_path, test_keys,
     eval_metric, dataset, model_kwargs, verbose) = args

    fold_dirs = _find_fold_dirs(split_path)
    if not fold_dirs:
        # legacy layout: epoch files directly in split_path
        epoch_files = _find_epoch_files(split_path)
        results: Dict[int, Tuple[float, float, float]] = {}
        for fname in epoch_files:
            epoch_num = int(re.findall(r'\d+', fname)[0])
            model = _load_model(split_path, fname, model_kwargs)
            fs, kt, sp, *_ = run_inference(
                model, dataset_path, test_keys,
                eval_metric, save_summary=False,
                dataset=dataset, verbose=verbose,
            )
            results[epoch_num] = (fs, kt, sp)
        best_epoch = max(results, key=lambda e: results[e][0]) if results else -1
        return split_id, best_epoch, results

    # new layout: aggregate over folds, key = epoch within fold dir
    # results keyed as (fold_idx, epoch_num) → (fs, kt, sp)
    fold_epoch_results: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    fold_best: Dict[int, Tuple[float, float, float]] = {}  # fold_idx → best metrics

    for fold_dir in fold_dirs:
        fold_idx = int(re.findall(r'\d+', fold_dir)[0])
        fold_path = join(split_path, fold_dir)
        epoch_files = _find_epoch_files(fold_path)
        fold_results: Dict[int, Tuple[float, float, float]] = {}

        for fname in epoch_files:
            epoch_num = int(re.findall(r'\d+', fname)[0])
            model = _load_model(fold_path, fname, model_kwargs)
            fs, kt, sp, *_ = run_inference(
                model, dataset_path, test_keys,
                eval_metric, save_summary=False,
                dataset=dataset, verbose=verbose,
            )
            fold_results[epoch_num] = (fs, kt, sp)
            fold_epoch_results[(fold_idx, epoch_num)] = (fs, kt, sp)

        if fold_results:
            best_ep = max(fold_results, key=lambda e: fold_results[e][0])
            fold_best[fold_idx] = fold_results[best_ep]

    # aggregate across folds: mean per epoch (averaged over all folds that have it)
    epoch_nums = sorted({ep for (_, ep) in fold_epoch_results})
    results: Dict[int, Tuple[float, float, float]] = {}
    for ep in epoch_nums:
        vals = [fold_epoch_results[(fi, ep)]
                for fi in range(1, len(fold_dirs) + 1)
                if (fi, ep) in fold_epoch_results]
        if vals:
            results[ep] = (
                float(np.nanmean([v[0] for v in vals])),
                float(np.nanmean([v[1] for v in vals])),
                float(np.nanmean([v[2] for v in vals])),
            )

    best_epoch = max(results, key=lambda e: results[e][0]) if results else -1
    return split_id, best_epoch, results


def _run_full_scan_parallel(
    split_ids: List[int],
    split_configs: dict,
    n_workers: int,
) -> Dict:
    n_workers = min(n_workers, len(split_ids))
    print(f'Full scan: {n_workers} parallel thread(s) across {len(split_ids)} splits\n')

    output: Dict = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_scan_split_worker, split_configs[s]): s
            for s in split_ids
        }
        for future in as_completed(futures):
            split_id = futures[future]
            try:
                sid, best_epoch, results = future.result()
                output[sid] = (best_epoch, results)
                best_fs = results[best_epoch][0]
                print(f'  Split {sid} done — best epoch: {best_epoch} (F1={best_fs:.2f}%)')
            except Exception as exc:
                logging.error(f'  Split {split_id} failed: {exc}', exc_info=True)

    return output


def _print_results(
    split_ids: List[int],
    best_epochs: Dict,
    split_results: Dict,
    best_avg_epoch: int,
    avg_fs: Dict,
    avg_ks: Dict,
    avg_ss: Dict,
) -> None:
    sep = '-' * 62
    print(f'\n{sep}')
    print(f"{'Split':<8} {'Epoch':>6}  {'F1 (%)':>8}  {'Kendall τ':>10}  {'Spearman ρ':>11}")
    print(sep)
    for s in split_ids:
        if s not in best_epochs:
            continue
        ep = best_epochs[s]
        fs, kt, sp = split_results[s]
        print(f'  {s:<6} {ep:>6}  {fs:>8.2f}  {kt:>10.4f}  {sp:>11.4f}')
    print(sep)
    print(
        f"  {'AVG':<6} {best_avg_epoch:>6}  "
        f'{avg_fs[best_avg_epoch]:>8.2f}  '
        f'{avg_ks[best_avg_epoch]:>10.4f}  '
        f'{avg_ss[best_avg_epoch]:>11.4f}'
    )
    print(f'{sep}\n')


def _save_xlsx(split_ids: List[int], all_epoch_results: Dict, dataset: str) -> None:
    from openpyxl.styles import PatternFill, Font
    from openpyxl.utils import get_column_letter

    split_ids_sorted = sorted(split_ids)
    n_splits = len(split_ids_sorted)
    all_epochs = sorted({ep for s in split_ids_sorted for ep in all_epoch_results.get(s, {})})

    n_present: List[int] = []
    avg_fs_row: List[float] = []
    avg_ks_row: List[float] = []
    avg_ss_row: List[float] = []

    for ep in all_epochs:
        present = [s for s in split_ids_sorted if ep in all_epoch_results.get(s, {})]
        n_present.append(len(present))
        if present:
            avg_fs_row.append(float(np.nanmean([all_epoch_results[s][ep][0] for s in present])))
            avg_ks_row.append(float(np.nanmean([all_epoch_results[s][ep][1] for s in present])))
            avg_ss_row.append(float(np.nanmean([all_epoch_results[s][ep][2] for s in present])))
        else:
            avg_fs_row.append(float('nan'))
            avg_ks_row.append(float('nan'))
            avg_ss_row.append(float('nan'))

    rows: Dict = {'Epoch': all_epochs}
    for s in split_ids_sorted:
        er = all_epoch_results.get(s, {})
        rows[f'F-score Split {s}']  = [er[ep][0] if ep in er else None for ep in all_epochs]
        rows[f'Kendall Split {s}']  = [er[ep][1] if ep in er else None for ep in all_epochs]
        rows[f'Spearman Split {s}'] = [er[ep][2] if ep in er else None for ep in all_epochs]

    rows['Avg F-score']  = avg_fs_row
    rows['Avg Kendall']  = avg_ks_row
    rows['Avg Spearman'] = avg_ss_row
    rows['N Splits']     = n_present

    df = pd.DataFrame(rows).set_index('Epoch')

    tuples = []
    for s in split_ids_sorted:
        for m in ('F-score', 'Kendall', 'Spearman'):
            tuples.append((f'Split {s}', m))
    for m in ('F-score', 'Kendall', 'Spearman'):
        tuples.append(('Average', m))
    tuples.append(('Average', 'N Splits'))
    df.columns = pd.MultiIndex.from_tuples(tuples)

    xlsx_path = f'{dataset}_epoch_metrics.xlsx'
    df.to_excel(xlsx_path)

    wb = load_workbook(xlsx_path)
    ws = wb.active

    ws.merge_cells('A1:A2')
    cell = ws['A1']
    cell.value = 'Epoch'
    cell.alignment = Alignment(horizontal='center', vertical='center')
    ws.delete_rows(3, 1)

    max_row = ws.max_row
    max_col = ws.max_column

    fill_full    = PatternFill('solid', start_color='C6EFCE')
    fill_partial = PatternFill('solid', start_color='FFEB9C')
    fill_missing = PatternFill('solid', start_color='FFC7CE')
    bold_font    = Font(bold=True)

    for row in ws.iter_rows(min_row=3, max_row=max_row):
        n_cell = row[max_col - 1]
        try:
            n = int(n_cell.value) if n_cell.value is not None else 0
        except (TypeError, ValueError):
            n = 0

        fill = fill_full if n == n_splits else (fill_partial if n > 1 else fill_missing)
        n_cell.fill = fill

        for cell in row[-(3 + 1):]:
            cell.font = bold_font

    for col_cells in ws.columns:
        length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
        ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(length + 2, 20)

    wb.save(xlsx_path)
    print(f'Full epoch metrics saved → {xlsx_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run inference and report best-epoch results for each split.'
    )
    parser.add_argument('--dataset', type=str, default='SumMe')
    parser.add_argument('--model_version', type=str, default='')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--save_summary', type=int, default=0)
    parser.add_argument('--save_results', type=int, default=0)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = vars(parser.parse_args())

    dataset = args['dataset']
    model_version = args['model_version']
    verbose = bool(args['verbose'])
    save_summary = bool(args['save_summary'])
    save_results = bool(args['save_results'])
    n_workers = args['workers']

    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'
    split_ids = list(range(5)) if dataset.lower() in ('summe', 'tvsum') else [0]

    model_kwargs = dict(
        input_size=1024,
        output_size=1024,
        num_segments=4,
        hidden_dim=args['hidden_dim'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
    )

    paths = get_paths(dataset)
    dataset_path = paths['dataset']
    split_file = paths['split']

    with open(split_file) as fp:
        split_data = json.load(fp)

    print(f'\nDataset: {dataset}  |  eval: {eval_metric}  |  splits: {split_ids}  |  device: {DEVICE}')

    best_epochs: Dict = {}
    split_results: Dict = {}
    all_epoch_results: Dict = {}

    if save_results:
        split_configs: Dict = {}
        for split_id in split_ids:
            split_path = f'Summaries/xLSTM/{dataset}{model_version}/models/split{split_id}'
            test_keys = (
                split_data[split_id]['test_keys']
                if isinstance(split_data, list)
                else split_data['test_keys']
            )
            fold_dirs = _find_fold_dirs(split_path)
            has_epochs = bool(_find_epoch_files(split_path))
            if not fold_dirs and not has_epochs:
                logging.warning(f'No fold dirs or epoch files in {split_path} — skipping split {split_id}')
                continue
            split_configs[split_id] = (
                split_id, split_path,
                dataset_path, test_keys,
                eval_metric, dataset, model_kwargs, verbose,
            )

        if not split_configs:
            print('No valid splits found. Check model paths.')
            return

        scan_output = _run_full_scan_parallel(list(split_configs.keys()), split_configs, n_workers)

        for sid, (best_epoch, results) in scan_output.items():
            best_epochs[sid] = best_epoch
            split_results[sid] = results[best_epoch]
            all_epoch_results[sid] = results

    else:
        for split_id in split_ids:
            split_path = f'Summaries/xLSTM/{dataset}{model_version}/models/split{split_id}'
            test_keys = (
                split_data[split_id]['test_keys']
                if isinstance(split_data, list)
                else split_data['test_keys']
            )

            # prefer split-level best_model.pkl (saved after all folds)
            split_best_pkl = join(split_path, 'best_model.pkl')
            if os.path.exists(split_best_pkl):
                model = _load_model(split_path, 'best_model.pkl', model_kwargs)
                best_epoch = -1
                print(f'Split {split_id}: using split best_model.pkl')
            else:
                # fall back to best fold best_model.pkl
                fold_dirs = _find_fold_dirs(split_path)
                fold_pkls = [(join(split_path, fd, 'best_model.pkl'), fd)
                             for fd in fold_dirs
                             if os.path.exists(join(split_path, fd, 'best_model.pkl'))]
                if fold_pkls:
                    # pick fold whose best_model.pkl gives best F-score
                    best_fs_fold, best_model, best_fold = -1.0, None, None
                    for pkl_path, fd in fold_pkls:
                        m = _load_model(os.path.dirname(pkl_path), 'best_model.pkl', model_kwargs)
                        fs_tmp, *_ = run_inference(m, dataset_path, test_keys,
                                                   eval_metric, False, dataset, verbose)
                        if fs_tmp > best_fs_fold:
                            best_fs_fold, best_model, best_fold = fs_tmp, m, fd
                    model = best_model
                    best_epoch = -1
                    print(f'Split {split_id}: using best_model.pkl from {best_fold}')
                else:
                    # legacy fallback: epoch files directly in split_path
                    epoch_files = _find_epoch_files(split_path)
                    if not epoch_files:
                        logging.warning(f'No checkpoints found in {split_path} — skipping split {split_id}')
                        continue
                    best_epoch = _load_best_epoch_from_fscores(split_path)
                    if best_epoch is not None:
                        fname = 'best_model.pkl' if os.path.exists(join(split_path, 'best_model.pkl')) else f'epoch-{best_epoch}.pkl'
                    else:
                        fname = epoch_files[-1]
                        best_epoch = int(re.findall(r'\d+', fname)[0])
                    model = _load_model(split_path, fname, model_kwargs)
                    print(f'Split {split_id}: legacy fallback epoch {best_epoch}')

            fs, kt, sp, *_ = run_inference(
                model, dataset_path, test_keys,
                eval_metric, save_summary, dataset, verbose,
            )
            best_epochs[split_id] = best_epoch
            split_results[split_id] = (fs, kt, sp)

    if not split_results:
        print('No results collected — check model paths and split files.')
        return

    valid_splits = list(split_results.keys())
    all_f = [split_results[s][0] for s in valid_splits]
    all_k = [split_results[s][1] for s in valid_splits]
    all_s = [split_results[s][2] for s in valid_splits]

    best_avg_epoch = best_epochs[max(valid_splits, key=lambda s: split_results[s][0])]
    avg_fs = {best_avg_epoch: float(np.nanmean(all_f))}
    avg_ks = {best_avg_epoch: float(np.nanmean(all_k))}
    avg_ss = {best_avg_epoch: float(np.nanmean(all_s))}

    _print_results(valid_splits, best_epochs, split_results, best_avg_epoch, avg_fs, avg_ks, avg_ss)

    if save_results:
        _save_xlsx(valid_splits, all_epoch_results, dataset)


if __name__ == '__main__':
    main()