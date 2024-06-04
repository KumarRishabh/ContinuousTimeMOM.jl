# write a slurm.sh file to run the test_branching_particle_filter.jl code 
# on compute canada cedar/graham cluster with 1 node and 4 cores 
# and 4 hours of time limit
# and 32 GB of memory 
# and 1 GPU
# and 1 job name
# and 1 output file
# and 1 error file
# and 1 email address
# and 1 email type


# write the slurm.sh file
echo '#!/bin/bash' > slurm.sh
echo '#SBATCH --account=def-vritsiou' >> slurm.sh
echo '#SBATCH --nodes=1' >> slurm.sh
echo '#SBATCH --ntasks=1' >> slurm.sh
echo '#SBATCH --cpus-per-task=4' >> slurm.sh
echo '#SBATCH --mem=32G' >> slurm.sh
echo '#SBATCH --time=4:00:00' >> slurm.sh
echo '#SBATCH --gres=gpu:1' >> slurm.sh
echo '#SBATCH --job-name=test_branching_particle_filter' >> slurm.sh
echo '#SBATCH --output=test_branching_particle_filter.out' >> slurm.sh
echo '#SBATCH --error=test_branching_particle_filter.err' >> slurm.sh

echo 'module load julia/1.10.0' >> slurm.sh
echo 'julia test_branching_particle_filter.jl' >> slurm.sh