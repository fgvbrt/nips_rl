# Description
[Reason8.ai](https://reason8.ai/) code for 3th place [NIPS learning to run challenge](https://www.crowdai.org/challenges/nips-2017-learning-to-run).

We are porting this code to pytorch [here](https://github.com/Scitator/Run-Skeleton-Run)

# Installation:
1) Get opensim package. You can use default package as described [here](https://github.com/stanfordnmbl/osim-rl) or
build by youself faster version [here](https://github.com/Scitator/opensim-core)
2) Run setup script:
    
        $ ./setup.sh

You may want to change conda env name in script and comment last line if not building opnesim by yourself 

3) If you experience theano errors try to move .theanorc file:

        $ mv ~/.theanorc ~/.theanorc.backup

# Running code
1) Activate environment:

        $ source activate nips_rl_fast3
2) Run code with best flags:

        $ python run_experiment.py --param_noise_prob 0.3 --flip_prob 1 --layer_norm

# Other
The final submitted model was trained in this [commit](https://github.com/fgvbrt/nips_rl/tree/e2ffeaa475c57c64bf6d4664b2ab47b46ecc1c6e/dpg3).

There are lot of branches with various ideas tested during competition but without documentation, you could check for example following branches:
    
   - [distributed ddpg with pyro4](https://github.com/fgvbrt/nips_rl/tree/farm/pyro_farm) inspired by [ctmarko repository](https://github.com/ctmakro/stanford-osrl)
   - [distributed CEM with pyro4](https://github.com/fgvbrt/nips_rl/tree/cem/pyro_farm) I am not sure that this is canonical implementation, it was done in the last night.
   - [we even tried to do planning](https://github.com/fgvbrt/nips_rl/tree/kr-uct/kr-uct) as described in [this article](https://www.ijcai.org/Proceedings/16/Papers/104.pdf) 

