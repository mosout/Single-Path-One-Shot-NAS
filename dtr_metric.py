import time


class TimeMetric():
    def __init__(self) -> None:
        self.started_at = 0.0
        self.count = 0
        self.all_time = 0.0
        self.do_count = False

    def reset(self):
        self.started_at = 0.0
        self.count = 0
        self.all_time = 0.0

    def start(self):
        self.started_at = time.time()

    def finish(self):
        if self.started_at == 0.0:
            raise RuntimeError("Please start first.")
        self.all_time += time.time()-self.started_at
        self.count += 1
        self.started_at = 0.0

    def get_average_time(self):
        if self.count == 0:
            raise RuntimeError("Please start and finish first.")
        return self.all_time / self.count

    def __call__(self, do_count: bool):
        self.do_count = do_count
        return self

    def __enter__(self):
        if self.do_count:
            self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.do_count:
            self.finish()


if __name__ == "__main__":
    m = TimeMetric()
    m.start()
    time.sleep(1)
    m.finish()
    m.start()
    time.sleep(1)
    m.finish()
    print(m.count)
    print(m.all_time)
