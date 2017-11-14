source activate nips_rl_fast3

1. Exclude params noise
python run_experiment.py --param_noise_prob 0. --flip_prob 1 --layer_norm --exp_name 101

2. Exclude layer norm
python run_experiment.py --param_noise_prob 0.3 --flip_prob 1 --exp_name 011

3. Exclude flipping
python run_experiment.py --param_noise_prob 0.3 --flip_prob 0 --layer_norm --exp_name 110

4. Add all
python run_experiment.py --param_noise_prob 0.3 --flip_prob 1 --layer_norm --exp_name 111