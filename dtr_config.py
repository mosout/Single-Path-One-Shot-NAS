import argparse


class Config:
    def __init__(self) -> None:
        self.num_warm_iters = 2
        self.device = "cuda+remat"
        # from args
        self.batch_size = 0
        self.num_iters = 0
        self.ddp = False
        self.print_mem_info = False
        self.dataset = "cifar10"
        self.classes = 10
        self.layers = 20
        self.num_choices = 4
        self.threshold = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("threshold", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("num_iters", type=int)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--print-mem-info", action="store_true")
    parser.add_argument("--device", type=str, default="cuda+remat")
    return parser.parse_args()


def get_config():
    config = Config()
    args = parse_args()
    # config.model_name = args.model_name
    config.threshold = args.threshold
    config.batch_size = args.batch_size
    config.num_iters = args.num_iters
    config.device = args.device
    config.print_mem_info = args.print_mem_info
    config.ddp = args.ddp
    return config
