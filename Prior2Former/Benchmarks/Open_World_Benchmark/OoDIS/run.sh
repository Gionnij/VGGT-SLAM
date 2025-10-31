cd ~/Benchmarks/OoDIS

t=-0.05
python run.py -o /Experiments/benchmark/Oodis/p2f_$t/LandF -t $t -v true
python scoring_program/evaluate.py /Experiments/benchmark/Oodis/p2f_$t/LandF/ /Datasets/dataset_LostAndFound/gtCoarse_Odis/train /Experiments/benchmark/Oodis/p2f_$t
