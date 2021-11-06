from typing import List


def get_multi_step_lr_with_warmup(
        optimizer,
        num_warmup_epochs: int,
        epoch_milestones: List[int],
        num_iters_per_epochs: int,
        gamma: float,
        last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    from torch.optim.lr_scheduler import LambdaLR
    from bisect import bisect_right

    num_warmup_steps = num_warmup_epochs * num_iters_per_epochs
    batch_milestones = [
        milestone * num_iters_per_epochs for milestone in epoch_milestones]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return gamma ** bisect_right(batch_milestones, current_step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == "__main__":
    import torch
    optimizer = torch.optim.SGD(torch.nn.Linear(10, 2).parameters(), lr=0.1)
    scheduler = get_multi_step_lr_with_warmup(
        optimizer, 1, [10, 20, 30], 5, 0.1)
    # print(optimizer.param_groups[0]['lr'])
    for epoch in range(50):
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch : {epoch} get_lr : {lr}')
        for batch in range(5):
            lr = optimizer.param_groups[0]['lr']
            print(f'batch : {batch} lr : {lr} get_lr : {scheduler.get_lr()}')
            scheduler.step()
    pass
