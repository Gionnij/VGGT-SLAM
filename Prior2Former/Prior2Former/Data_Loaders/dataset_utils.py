from collections import defaultdict
from itertools import product


def create_dataset(name,size,ds_version,preprocessing,augmentation):
    ## name - 0/1 - 0/1 - 0/1 -0/2
    combinations=list(product([name],size.values(),ds_version,preprocessing.values(),augmentation))
    datasets=[]
    for comb in combinations:
        datasets.append("".join(comb))
    dataset_sizes = defaultdict(list)
    dataset_preprocessing = defaultdict(list)
    for k,v in size.items():
        for data in datasets:
            if v in data:
                dataset_sizes[k].append(data)
    for k, v in preprocessing.items():
        for data in datasets:
            if v in data:
                dataset_preprocessing[k].append(data)
    return datasets, dataset_sizes, dataset_preprocessing,
