python -m model.main --split_index 0 --n_epochs 200 --batch_size 20 --video_type 'SumMe'
python -m model.main --split_index 1 --n_epochs 200 --batch_size 20 --video_type 'SumMe'
python -m model.main --split_index 2 --n_epochs 200 --batch_size 20 --video_type 'SumMe'
python -m model.main --split_index 3 --n_epochs 200 --batch_size 20 --video_type 'SumMe'
python -m model.main --split_index 4 --n_epochs 200 --batch_size 20 --video_type 'SumMe'
python -m model.main --split_index 0 --n_epochs 200 --batch_size 40 --video_type 'TVSum'
python -m model.main --split_index 1 --n_epochs 200 --batch_size 40 --video_type 'TVSum'
python -m model.main --split_index 2 --n_epochs 200 --batch_size 40 --video_type 'TVSum'
python -m model.main --split_index 3 --n_epochs 200 --batch_size 40 --video_type 'TVSum'
python -m model.main --split_index 4 --n_epochs 200 --batch_size 40 --video_type 'TVSum'
python -m inference.inference --save_results 1 --workers 5
python -m inference.inference --dataset 'TVSum'  --save_results 1 --workers 5
#python -m inference.plot_video_scores --dataset SumMe --all_splits --epoch 101 --output_dir plots
#python -m inference.plot_video_scores --dataset TVSum --all_splits --epoch 199 --output_dir plots