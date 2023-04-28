import os
import torch
import torch.nn as nn
import json
import utils
from dtr_config import get_config
from dtr_metric import TimeMetric
from dtr_profile_util import ProfileGuard
from dtr_dataset import ImagenetStyleDataset
from models.model import SinglePath_OneShot

def sync():
    torch.comm.barrier()


if __name__ == "__main__":
    cfg = get_config()
    torch.remat.set_budget(f"{cfg.threshold}MB")
    torch.remat.set_small_pieces_optimization(False)
    model = SinglePath_OneShot(cfg.dataset, False, cfg.classes, cfg.layers)
    # logging.info(model)
    model = model.to(cfg.device)
    criterion = nn.CrossEntropyLoss().to(cfg.device)

    dataset = ImagenetStyleDataset(
        batch_size=cfg.batch_size, length=32, device=cfg.device
    )
    train_data, train_label = dataset[0]
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), 0.025, 0.9, 3e-4)
    m = TimeMetric()
    profiler_guard = ProfileGuard()
    with profiler_guard:
        for i in range(cfg.num_iters):
            with m(i >= cfg.num_warm_iters):
                # print(
                #     f"iter: {i} start, device: {cfg.device}, mem: {torch._oneflow_internal.dtr.allocated_memory(cfg.device)}"
                # )
                choice = utils.random_choice(cfg.num_choices, cfg.layers)
                logits = model(train_data, choice)
                loss = criterion(logits, train_label)
                del logits
                loss.backward()
                del loss
                optimizer.step()
                optimizer.zero_grad()
                sync()
    if cfg.num_iters > cfg.num_warm_iters:
        result_json_path = os.getenv("ONEFLOW_DTR_SUMMARY_FILE_PREFIX")
        if result_json_path is not None:
            with open("result_json_path", "w") as f:
                json.dump({"real time": m.get_average_time()}, f)
        print(f"{m.count} iters: avg {m.get_average_time()}s")
