Установка:
1) Скопировать директорию opensim в проект
2) ./setup.sh # установить все пакеты
3) mv ~/.theanorc ~/..theanorc.backup # забекапить theano файл с настройками


Запуск:
всего 4 эксперимента. Перед запуском каждого эксперимента надо активировать среду conda
source activate nips_rl_fast3
Для каждого процесса количество тредов задается флагом n_threads

1. Exclude params noise
python run_experiment.py --param_noise_prob 0. --flip_prob 1 --layer_norm --exp_name 101 --n_threads n_threads

2. Exclude layer norm
python run_experiment.py --param_noise_prob 0.3 --flip_prob 1 --exp_name 011 --n_threads n_threads

3. Exclude flipping
python run_experiment.py --param_noise_prob 0.3 --flip_prob 0 --layer_norm --exp_name 110 --n_threads n_threads

4. Add all
python run_experiment.py --param_noise_prob 0.3 --flip_prob 1 --layer_norm --exp_name 111 --n_threads n_threads