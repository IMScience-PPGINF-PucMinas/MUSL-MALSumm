import logging

from configs.constants import (
    SUMME_DATASET_PATH, SUMME_SPLIT_FILE_PATH, SUMME_SAVE_DIR, SUMME_LOG_DIR, SUMME_SCORE_DIR,
    TVSUM_DATASET_PATH, TVSUM_SPLIT_FILE_PATH, TVSUM_SAVE_DIR, TVSUM_LOG_DIR, TVSUM_SCORE_DIR,
    MRHISUM_DATASET_PATH, MRHISUM_SPLIT_FILE_PATH, MRHISUM_SAVE_DIR, MRHISUM_LOG_DIR, MRHISUM_SCORE_DIR,
)

_DATASET_PATHS = {
    'summe': {
        'dataset': SUMME_DATASET_PATH,
        'split': SUMME_SPLIT_FILE_PATH,
        'save_dir': SUMME_SAVE_DIR,
        'log_dir': SUMME_LOG_DIR,
        'score_dir': SUMME_SCORE_DIR,
    },
    'tvsum': {
        'dataset': TVSUM_DATASET_PATH,
        'split': TVSUM_SPLIT_FILE_PATH,
        'save_dir': TVSUM_SAVE_DIR,
        'log_dir': TVSUM_LOG_DIR,
        'score_dir': TVSUM_SCORE_DIR,
    },
    'mrhisum': {
        'dataset': MRHISUM_DATASET_PATH,
        'split': MRHISUM_SPLIT_FILE_PATH,
        'save_dir': MRHISUM_SAVE_DIR,
        'log_dir': MRHISUM_LOG_DIR,
        'score_dir': MRHISUM_SCORE_DIR,
    },
}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def get_paths(dataset_name: str) -> dict:
    key = dataset_name.lower()
    if key not in _DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Valid options: {list(_DATASET_PATHS)}"
        )
    return _DATASET_PATHS[key]