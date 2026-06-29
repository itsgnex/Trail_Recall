import time


class DwellTrigger:
    def __init__(self, dwell_seconds, cooldown_seconds):
        self.dwell_seconds = dwell_seconds
        self.cooldown_seconds = cooldown_seconds
        self.kind = "other"
        self.since = 0.0
        self.last_analysis = 0.0
        self.cooldown_until = 0.0

    def ready_to_analyze(self, interval):
        now = time.monotonic()
        if now - self.last_analysis < interval:
            return False
        self.last_analysis = now
        return True

    def update(self, kind):
        now = time.monotonic()
        if now < self.cooldown_until or kind == "other":
            if kind == "other":
                self.kind = "other"
                self.since = 0.0
            return False

        if kind != self.kind:
            self.kind = kind
            self.since = now
            return False

        if now - self.since >= self.dwell_seconds:
            self.cooldown_until = now + self.cooldown_seconds
            self.kind = "other"
            self.since = 0.0
            return True
        return False


def _demo():
    trigger = DwellTrigger(0.01, 0.05)
    assert not trigger.update("plant")
    time.sleep(0.02)
    assert trigger.update("plant")
    assert not trigger.update("plant")


if __name__ == "__main__":
    _demo()
