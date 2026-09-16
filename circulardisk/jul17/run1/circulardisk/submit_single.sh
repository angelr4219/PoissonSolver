#### submit_arrayjob.sh START ####
#!/bin/bash
#$ -wd /u/home/a/angelr19/runs/circulardisk/
# error = Merged with joblog
##
#$ -j y
## Edit the line below as needed:
#$ -l highp,h_rt=8:00:00,h_data=7G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 4
# Email address to notify
#$ -M $angelr19@g.ucla.edu
# Notify when
#$ -m ea
#$ -t 1-1:1

# echo job info on joblog:
echo "Job $JOB_ID.$SGE_TASK_ID started on:   " `hostname -s`
echo "Job $JOB_ID.$SGE_TASK_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:
module load gcc/10.2.0
module load /u/local/Modules/modulefiles/python/3.7.3
module load intel

## substitute the command to run your code
## in the two lines below:
echo '/u/home/c/cander/MaSQE/bin/RunMDMM -f dotarray1disk2.xml'
pwd
/u/home/c/cander/MaSQE/bin/RunMDMM -f dotarray1disk2.xml

# echo job info on joblog:
echo "Job $JOB_ID.$SGE_TASK_ID ended on:   " `hostname -s`
echo "Job $JOB_ID.$SGE_TASK_ID ended on:   " `date `
echo " "
#### submit_arrayjob.sh STOP ####