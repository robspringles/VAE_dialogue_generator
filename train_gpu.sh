#srun --mem 20G --gres=gpu --partition=amd-longq --nodes 1 --mail-type=ALL sh train.sh
srun --mem 20G --partition intel-longq --gres=gpu --nodelist=dgx01 sh train.sh
