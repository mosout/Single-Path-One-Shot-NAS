import os
import json
import torch as flow


def update_dataset(profiler, show_events: bool = False):
    dataset_file_path = os.getenv("ONEFLOW_DTR_OP_TIME_DATASET")
    if dataset_file_path is None:
        raise RuntimeError(
            "Please specify the path of the time dataset with `ONEFLOW_DTR_OP_TIME_DATASET=/path/to/dataset`")
    if os.path.exists(dataset_file_path):
        with open(dataset_file_path, "r") as f:
            time_dict = json.load(f)
    else:
        time_dict = {}
    events = profiler.key_averages(
        group_by_input_shape=True, group_by_attributes=True)
    if show_events:
        print(events)
    new_time_dict = {}
    for e in events:
        if isinstance(e, flow.profiler.events.KernelEvent):
            # new_time_dict[
            #     f"{e.name} {e.description['shape']} {e.description['attr']}"
            # ] = (e.cuda_time_total, e.count)
            new_time_dict[
                f"{e.name} {e.description['input_shapes'][0]} {e.description['attrs'][0]}"
            ] = e.cuda_time

    time_dict.update(new_time_dict)

    with open(dataset_file_path, "w") as f:
        json.dump(time_dict, f)


class ProfileGuard:
    def __init__(self) -> None:
        do_profile = os.getenv("ENABLE_PROFILE_FOR_DTR")
        if do_profile is None:
            raise RuntimeError(
                "Please specify the env var `ENABLE_PROFILE_FOR_DTR=0/1`")
        if do_profile == "1":
            self.profiler = flow.profiler.profile(
                record_shapes=True, record_attrs=True)
        else:
            self.profiler = None

    def __enter__(self):
        if self.profiler is not None:
            self.profiler.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
            update_dataset(self.profiler, True)
