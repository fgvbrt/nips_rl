Установка:
1) Скопировать директорию opensim в проект
2) ./setup.sh # установить все пакеты
3) mv ~/.theanorc ~/..theanorc.backup # забекапить theano файл с настройками


Запуск:
всего 4 эксперимента. Перед запуском каждого эксперимента надо активировать среду conda
source activate nips_rl_fast3
Для каждого процесса количество тредов задается флагом num_agents

1. Exclude params noise
python run_experiment.py --param_noise_prob 0. --flip_prob 1 --layer_norm --exp_name 101 --num_agents num_agents

2. Exclude layer norm
python run_experiment.py --param_noise_prob 0.3 --flip_prob 1 --exp_name 011 --num_agents num_agents

3. Exclude flipping
python run_experiment.py --param_noise_prob 0.3 --flip_prob 0 --layer_norm --exp_name 110 --num_agents num_agents

4. Add all
python run_experiment.py --param_noise_prob 0.3 --flip_prob 1 --layer_norm --exp_name 111 --num_agents num_agents