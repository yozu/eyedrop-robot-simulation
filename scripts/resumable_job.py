"""Mac上の単一実験を排他実行し、SSH切断後も結果を保存する。"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

SCHEMA_VERSION = 1
POLL_SECONDS = 0.2
TERMINATE_GRACE_SECONDS = 5
DESCENDANT_SCAN_SECONDS = 1
TIMEOUT_EXIT_CODE = 124
SIGNAL_EXIT_BASE = 128
INCOMPLETE_EXIT_CODE = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def event(root, value):
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": now(), **value}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def lock_file(root):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "execution.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("別の実験が排他実行中です") from None
    return handle


def process_info(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None



def record_descendants(directory,pid):
    path=Path(directory)/'descendants.json'
    known=json.loads(path.read_text()) if path.exists() else []
    keys={(row['pid'],row['process_identity']) for row in known}
    listing=subprocess.run(['ps','-axo','pid=,ppid=,stat=,lstart=,command='],capture_output=True,text=True,check=True)
    rows=[]
    for line in listing.stdout.splitlines():
        fields=line.split(None,8)
        if len(fields)==9 and not fields[2].startswith('Z'):
            rows.append({'pid':int(fields[0]),'parent':int(fields[1]),'process_identity':' '.join(fields[3:])})
    parents={pid}
    while True:
        children=[row for row in rows if row['parent'] in parents and row['pid'] not in parents]
        if not children:break
        for row in children:
            parents.add(row['pid']);key=(row['pid'],row['process_identity'])
            if key not in keys:known.append(row);keys.add(key)
    if not path.exists() or len(known)!=len(json.loads(path.read_text())):atomic_json(path,known)


def live_descendants(directory):
    path=Path(directory)/'descendants.json'
    if not path.exists():return []
    alive=[]
    for row in json.loads(path.read_text()):
        identity=process_info(row['pid'])
        if identity and ' '.join(identity.split())==' '.join(row['process_identity'].split()):alive.append(row)
    return alive


def status(root):
    root = Path(root)
    pointer = root / "current.json"
    current = json.loads(pointer.read_text()) if pointer.exists() else None
    try:
        handle = lock_file(root)
    except RuntimeError:
        locked = True
    else:
        locked = False
        handle.close()
    if current:
        current["live_process"] = process_info(current["pid"])
        current["identity_matches"] = bool(current["live_process"]) and current["live_process"] == current["process_identity"]
        receipt = root / current["id"] / "exit.json"
        current["exit"] = json.loads(receipt.read_text()) if receipt.exists() else None
        current["live_descendants"] = live_descendants(root/current["id"])
    return {"locked": locked, "current": current}


def reconcile(root, proof):
    root=Path(root).resolve();proof=Path(proof).resolve()
    if not proof.is_file():raise ValueError('照合記録が必要です')
    with lock_file(root):
        current=json.loads((root/'current.json').read_text())
        directory=root/current['id']
        if (directory/'exit.json').exists():raise RuntimeError('終了記録は上書きしません')
        members=subprocess.run(['ps','-axo','pid=,pgid=,stat=,command='],capture_output=True,text=True,check=True)
        for line in members.stdout.splitlines():
            fields=line.split(None,3)
            if len(fields)>=3 and fields[1]==str(current['pid']) and not fields[2].startswith('Z'):
                raise RuntimeError('実験のプロセス群が残っています')
        if live_descendants(directory):raise RuntimeError('別セッションの子プロセスが残っています')
        runner_identity=process_info(current['runner_pid'])
        if runner_identity and (not current.get('runner_process_identity') or runner_identity==current['runner_process_identity']):raise RuntimeError('実行親が存在します。自動解除しません')
        result={'id':current['id'],'finished_at':now(),'returncode':None,'reason':'interrupted_reconciled',
                'proof':str(proof),'proof_sha256':hashlib.sha256(proof.read_bytes()).hexdigest(),
                'note':'元の終了コードは不明。性能の合否は照合記録と段階判定で別に決める'}
        atomic_json(directory/'exit.json',result);event(root,{'kind':'reconciled',**result})
        return result


def run_job(root, run_id, command, timeout):
    root = Path(root).resolve()
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("実験IDは単一のディレクトリ名で指定してください")
    if timeout <= 0 or not command:
        raise ValueError("有限の実行時間とコマンドが必要です")
    with lock_file(root):
        pointer = root / "current.json"
        if pointer.exists():
            previous = json.loads(pointer.read_text())
            receipt = root / previous["id"] / "exit.json"
            if not receipt.exists():
                raise RuntimeError("前の実験に終了記録がありません。子プロセスと出力の手動照合が必要です")
            if live_descendants(root/previous["id"]):raise RuntimeError("前の実験の子プロセスが残っています")
        directory = root / run_id
        directory.mkdir(exist_ok=False)
        started = time.monotonic()
        with (directory / "stdout.log").open("wb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            record = {"schema": SCHEMA_VERSION, "id": run_id, "pid": child.pid,
                      "runner_pid": os.getpid(), "runner_process_identity": process_info(os.getpid()),
                      "process_identity": process_info(child.pid),
                      "command": command, "command_sha256": hashlib.sha256(
                          json.dumps(command).encode()).hexdigest(),
                      "started_at": now(), "timeout_seconds": timeout}
            atomic_json(directory / "started.json", record)
            atomic_json(pointer, record)
            event(root, {"kind": "started", **record})
            reason = "exited"
            last_scan = 0
            try:
                while child.poll() is None:
                    if time.monotonic()-last_scan>=DESCENDANT_SCAN_SECONDS:
                        record_descendants(directory,child.pid);last_scan=time.monotonic()
                    if time.monotonic() - started >= timeout:
                        reason = "timeout"
                        break
                    time.sleep(POLL_SECONDS)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=TERMINATE_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                remaining=live_descendants(directory)
                if remaining and reason=="exited":reason="descendants_remaining"
                result = {"id": run_id, "finished_at": now(), "live_descendants":remaining, "returncode": child.returncode,
                          "reason": reason, "elapsed_seconds": time.monotonic() - started}
                atomic_json(directory / "exit.json", result)
                event(root, {"kind": "finished", **result})
        return result


def result_exit_code(result):
    if result['reason'] == 'timeout':
        return TIMEOUT_EXIT_CODE
    if result['reason'] != 'exited' or result['returncode'] is None:
        return INCOMPLETE_EXIT_CODE
    code = result['returncode']
    return SIGNAL_EXIT_BASE - code if code < 0 else code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["status", "run", "reconcile"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--id")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--proof")
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1:]
    if args.operation=='status':result=status(args.root)
    elif args.operation=='reconcile':result=reconcile(args.root,args.proof)
    else:result=run_job(args.root,args.id,command,args.timeout)
    print(json.dumps(result, ensure_ascii=False))
    return result_exit_code(result) if args.operation == 'run' else 0


if __name__ == "__main__":
    sys.exit(main())
