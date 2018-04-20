#srun --gres=gpu --partition=amd-longq --nodes 1 --mail-user=xx6@hw.ac.uk --mail-type=ALL sh translate.sh
srun --partition intel-longq --gres=gpu --nodelist=dgx01 sh translate.sh
