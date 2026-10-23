"""Single-process launch, and stopping a training run at a PPO update boundary."""
from contextlib import AbstractContextManager
import math
import os
import signal
import time


def require_single_process():
    for name, expected in (('WORLD_SIZE', 1), ('LOCAL_WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0)):
        raw = os.environ.get(name, str(expected))
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f'{name} must be an integer; this entry supports single-process PPO only') from None
        if value != expected:
            raise ValueError(f'Multi-GPU joint PPO training is not integrated: {name}={raw}. '
                             'Use the single-process entry; distributed launching is deferred.')


class BoundaryStop(AbstractContextManager):
    """Signals request a stop; saving happens in normal code after a whole update.

    This is a soft deadline. A long physics/update/save operation cannot be
    preempted safely; the external scheduler still owns its hard deadline.
    """
    def __init__(self, max_seconds=None, margin_seconds=30., *, started=None, clock=None):
        if max_seconds is not None and (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
                                        or not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('max_seconds must be finite and positive')
        if (isinstance(margin_seconds, bool) or not isinstance(margin_seconds, (int, float))
                or not math.isfinite(margin_seconds) or margin_seconds < 0):
            raise ValueError('stop_margin_seconds must be finite and nonnegative')
        if max_seconds is not None and margin_seconds >= max_seconds:
            raise ValueError('stop_margin_seconds must be less than max_seconds')
        self.clock = clock or time.monotonic
        self.started = self.clock() if started is None else started
        self.max_seconds, self.margin_seconds = max_seconds, margin_seconds
        self.signal_name, self._previous = None, {}

    def _request(self, number, frame):
        if self.signal_name is None:
            self.signal_name = signal.Signals(number).name

    def reason(self):
        if self.signal_name is not None:
            return 'signal:' + self.signal_name
        if self.max_seconds is not None and self.clock() - self.started >= self.max_seconds - self.margin_seconds:
            return 'wall_time_budget'
        return None

    def __enter__(self):
        try:
            for name in ('SIGTERM', 'SIGINT', 'SIGUSR1'):
                number = getattr(signal, name, None)
                if number is not None:
                    self._previous[number] = signal.signal(number, self._request)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for number, handler in self._previous.items():
            signal.signal(number, handler)
        self._previous.clear()
