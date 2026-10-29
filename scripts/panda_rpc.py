"""Mac内の描画プロセスとの行単位通信。正解情報は教師・採点側だけで扱う。"""
import json
import os
from pathlib import Path
import subprocess


class PandaBrowser:
    def __init__(self, log, *, person='original', hybrid_rgb=False):
        import pybullet_data
        root = Path(__file__).resolve().parents[1]
        if not isinstance(person, str): raise ValueError('人物IDは文字列で指定してください')
        environment = {**os.environ, "PANDA_ASSETS": str(Path(pybullet_data.getDataPath()) / "franka_panda"), "PANDA_PERSON": person}
        environment['PANDA_HYBRID_RGB'] = '1' if hybrid_rgb else '0'
        self.log = open(log, "w")
        self.process = subprocess.Popen(["/opt/homebrew/bin/node", str(root / "scripts/panda_browser.cjs")],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, env=environment)
        response = self.process.stdout.readline()
        ready=json.loads(response) if response else {}
        if not ready.get("ready"):
            raise RuntimeError("Mac描画の起動失敗。browser.logを確認してください")
        if ready.get('person')!=person:
            self.close()
            raise RuntimeError('要求した人物と描画人物が一致しません')
        self.person=person

    def call(self, operation, **arguments):
        self.process.stdin.write(json.dumps({"operation": operation, **arguments}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("描画プロセスが終了しました")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result")

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"operation":"close"}\n')
            self.process.stdin.flush()
            self.process.stdin.close()
            self.process.wait(timeout=15)
        self.log.close()
