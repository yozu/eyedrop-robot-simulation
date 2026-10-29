"""同時処理1件、待機は最新1件。物理更新のスレッドを待たせない。"""
import threading

SHUTDOWN_TIMEOUT = 3.0


class LatestWorker:
    def __init__(self, process):
        self.process = process
        self.condition = threading.Condition()
        self.pending = None
        self.result = None
        self.busy = False
        self.closed = False
        self.error = None
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def submit(self, value):
        with self.condition:
            if self.closed or self.error is not None:
                raise RuntimeError("停止済みの送受信担当には送れません")
            if self.pending is not None:
                self.dropped += 1
            self.pending = value
            self.condition.notify_all()

    def take(self):
        with self.condition:
            result, self.result = self.result, None
            return result

    def wait_idle(self, timeout):
        with self.condition:
            return self.condition.wait_for(
                lambda: not self.busy and self.pending is None, timeout)

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                value, self.pending = self.pending, None
                self.busy = True
            try:
                result = self.process(value)
            except Exception as exc:
                with self.condition:
                    self.error = exc
                    self.pending = None
                    self.result = None
                    self.busy = False
                    self.closed = True
                    self.condition.notify_all()
                return
            with self.condition:
                self.result = result
                self.busy = False
                self.condition.notify_all()

    def __exit__(self, *_):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(SHUTDOWN_TIMEOUT)
        if self.thread.is_alive():
            raise TimeoutError("送受信担当が停止しません。I/Oタイムアウトを確認してください")
