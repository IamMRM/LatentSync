import os
import subprocess
import signal
import sys
import time
import torch

WORKER_PORT_START = 7860  # first Gradio port


def start_worker(gpu_id: int):
    """Launch gradio_app.py on the given GPU and port."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    port = WORKER_PORT_START + gpu_id

    cmd = [
        sys.executable,
        "gradio_app.py",
        "--server_port",
        str(port),
        "--server_name",
        "0.0.0.0",
    ]
    print(f"🚀 Launching worker on GPU {gpu_id} → http://localhost:{port}")
    return subprocess.Popen(cmd, env=env)


def main():
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("❌ No CUDA devices found – cannot start multi-GPU server.")
        sys.exit(1)

    workers = [start_worker(i) for i in range(gpu_count)]
    print(f"✅ Started {gpu_count} workers. Press Ctrl-C to stop.")

    try:
        # Wait indefinitely, forward signals to children
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹  Stopping workers …")
        for p in workers:
            p.send_signal(signal.SIGINT)
        for p in workers:
            p.wait()
        print("✅ All workers stopped.")


if __name__ == "__main__":
    main() 