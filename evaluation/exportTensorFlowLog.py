import csv
import os
import sys
import time

try:
    import collections.abc
except ImportError:
    import collections

_event_accumulator = None

for _import in [
    'tensorflow.python.summary',
    'tensorflow.tensorboard.backend.event_processing',
    'tensorboard.backend.event_processing',
]:
    try:
        import importlib
        _mod = importlib.import_module(f'{_import}.event_accumulator')
        _event_accumulator = _mod.event_accumulator
        break
    except ImportError:
        continue

if _event_accumulator is None:
    raise ImportError('Could not locate and import Tensorflow event accumulator.')

event_accumulator = _event_accumulator

SUMMARIES_DEFAULT = ['scalars']


class Timer:
    def __init__(self, name=None):
        self.name = name
        self.tStart = None

    def __enter__(self):
        self.tStart = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name:
            print(f'[{self.name}]')
        print(f'Elapsed: {time.time() - self.tStart:.4f}s')


def _exit_with_usage() -> None:
    print('\nUsage:')
    print('   python exportTensorFlowLog.py <input-path-to-logfile> <output-folder> [<summaries>]')
    print('\nInputs:')
    print('   <input-path-to-logfile>  - Path to TensorFlow logfile.')
    print('   <output-folder>          - Path to output folder.')
    print(
        '   <summaries>              - (Optional) Comma-separated list of summaries. '
        f'Default: {", ".join(SUMMARIES_DEFAULT)}'
    )
    sys.exit(1)


def _parse_args():
    if len(sys.argv) < 3:
        _exit_with_usage()

    input_log = sys.argv[1]
    output_folder = sys.argv[2]
    summaries = (
        SUMMARIES_DEFAULT
        if len(sys.argv) < 4 or sys.argv[3] == 'all'
        else sys.argv[3].split(',')
    )

    if any(s not in SUMMARIES_DEFAULT for s in summaries):
        print('Unknown summary! See usage for acceptable summaries.')
        _exit_with_usage()

    return input_log, output_folder, summaries


def export_scalars(ea, output_folder: str, tags: dict) -> None:
    csv_path = os.path.join(output_folder, 'scalars.csv')
    print(f'CSV-path: {csv_path}')
    scalar_tags = tags['scalars']

    with Timer():
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['wall_time', 'step'] + scalar_tags)

            base_vals = ea.Scalars(scalar_tags[0])
            for i in range(len(base_vals)):
                v = base_vals[i]
                row = [v.wall_time, v.step]
                for tag in scalar_tags:
                    row.append(ea.Scalars(tag)[i].value)
                writer.writerow(row)


def export_images(ea, output_folder: str, tags: dict) -> None:
    image_dir = os.path.join(output_folder, 'images')
    print(f'Image dir: {image_dir}')
    with Timer():
        for image_tag in tags['images']:
            tag_dir = os.path.join(image_dir, image_tag)
            os.makedirs(tag_dir, exist_ok=True)
            for image in ea.Images(image_tag):
                img_path = os.path.join(tag_dir, f'{image.step}.png')
                with open(img_path, 'wb') as f:
                    f.write(image.encoded_image_string)


def main() -> None:
    input_log, output_folder, summaries = _parse_args()

    with Timer():
        ea = event_accumulator.EventAccumulator(
            input_log,
            size_guidance={
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 0,
            },
        )

    with Timer():
        ea.Reload()

    tags = ea.Tags()
    os.makedirs(output_folder, exist_ok=True)

    if 'scalars' in summaries:
        export_scalars(ea, output_folder, tags)

    if 'images' in summaries:
        export_images(ea, output_folder, tags)

    for unsupported in ('audio', 'compressedHistograms', 'histograms'):
        if unsupported in summaries:
            print(f'\n{unsupported} export is not yet supported.')


if __name__ == '__main__':
    main()