# Validation 
## Baseline (Instruction Tuning)
for f in scripts/iclr2025/expt1/res/finetune_{5,7,10,15,20}e.sh; do sbatch --account nlp --partition sphinx --nodelist sphinx4    --gres gpu:2 --mem 50G --exclude jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31    $f; done
## Response Tuning
for f in scripts/iclr2025/expt1/ins/finetune_{5,7,10,15,20}e.sh; do sbatch --account nlp --partition sphinx --nodelist sphinx4    --gres gpu:2 --mem 50G --exclude jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31    $f; done

# Test
# Baseline seeds
for seed in 1 2 3 4 5 ; do sbatch --account nlp --partition sphinx --nodelist sphinx4    --gres gpu:2 --mem 50G --exclude jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31  scripts/iclr2025/expt1/ins/finetune_seed${seed}.sh; done
# Response tuning seeds
for seed in 1 2 3 4 5 ; do sbatch --account nlp --partition sphinx --nodelist sphinx4    --gres gpu:2 --mem 50G --exclude jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31  scripts/iclr2025/expt1/res/finetune_seed${seed}.sh; done
