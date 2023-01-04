import torch
from . import samplers

def make_data_sampler(dataset, shuffle, distributed, is_train=True):
    # Only do weighted sampling for training
    if distributed:
        # if is_train:
        #     return samplers.DistributedWeightedSampler(dataset, shuffle=shuffle)
        # else:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights, num_samples=len(dataset))
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_batch_data_sampler(dataset,
                            sampler,
                            aspect_grouping,
                            batch_per_gpu,
                            max_iters=None,
                            start_iter=0,
                            dataset_name=None):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset, dataset_name=dataset_name)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, batch_per_gpu, drop_uneven=False)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_per_gpu, drop_last=False)
    if max_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iters, start_iter)
    return batch_sampler