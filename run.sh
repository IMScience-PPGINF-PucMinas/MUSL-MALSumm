python -m model.main --split_index 0 --n_epochs 200 --batch_size 20 --video_type 'SumMe' --alpha_rl 0.0 --lambda_rank 0.1
python -m model.main --split_index 1 --n_epochs 200 --batch_size 20 --video_type 'SumMe' --alpha_rl 0.0 --lambda_rank 0.1
python -m model.main --split_index 2 --n_epochs 200 --batch_size 20 --video_type 'SumMe' --alpha_rl 0.0 --lambda_rank 0.1
python -m model.main --split_index 3 --n_epochs 200 --batch_size 20 --video_type 'SumMe' --alpha_rl 0.0 --lambda_rank 0.1
python -m model.main --split_index 4 --n_epochs 200 --batch_size 20 --video_type 'SumMe' --alpha_rl 0.0 --lambda_rank 0.1
python -m inference.inference --best_fscore_only False

