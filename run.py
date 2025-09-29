import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
WEB_DIR = ROOT / "web"

# Common entrypoints we’ll auto-discover if env/CLI not provided
BACKEND_CANDIDATES = ("ai_service.py", "backend_service.py", "server.py", "main.py", "app.py")
WEB_CANDIDATES = ("web_app.py", "app.py", "main.py", "server.py")


def _find_script(base: Path, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        p = base / name
        if p.is_file():
            return str(p.name)  # return just the script name; we cd into base
    return None


def _resolve_scripts(cli_backend: Optional[str], cli_web: Optional[str]) -> Tuple[Path, str, Path, str]:
    # Env overrides
    env_backend = os.getenv("RUN_BACKEND")
    env_web = os.getenv("RUN_WEB")

    backend_dir = BACKEND_DIR if BACKEND_DIR.is_dir() else ROOT
    web_dir = WEB_DIR if WEB_DIR.is_dir() else ROOT

    backend_script = cli_backend or env_backend or _find_script(backend_dir, BACKEND_CANDIDATES)
    web_script = cli_web or env_web or _find_script(web_dir, WEB_CANDIDATES)

    if not backend_script:
        raise FileNotFoundError(
            f"Could not find a backend entrypoint in {backend_dir}. "
            f"Tried: {', '.join(BACKEND_CANDIDATES)}. "
            f"Set RUN_BACKEND or pass --backend."
        )
    if not web_script:
        raise FileNotFoundError(
            f"Could not find a web entrypoint in {web_dir}. "
            f"Tried: {', '.join(WEB_CANDIDATES)}. "
            f"Set RUN_WEB or pass --web."
        )

    return backend_dir, backend_script, web_dir, web_script


def launch(title: str, script: str, cwd: Path) -> subprocess.Popen:
    """
    Launch `python <script>` as a real child process we can track.
    On Windows, give it its own console & process group so we can send CTRL+BREAK.
    """
    python_exe = sys.executable
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        # NOTE: CREATE_NEW_PROCESS_GROUP is required to allow CTRL_BREAK_EVENT.
        return subprocess.Popen(
            [python_exe, script],
            cwd=str(cwd),
            creationflags=creationflags,
            close_fds=False,
        )
    else:
        # On Unix, inherit the same terminal. You can detach if desired.
        return subprocess.Popen(
            [python_exe, script],
            cwd=str(cwd),
            close_fds=True,
        )


def graceful_terminate(p: subprocess.Popen, name: str, timeout: float = 8.0):
    if p is None or p.poll() is not None:
        return
    try:
        if os.name == "nt":
            # Send CTRL+BREAK to the process group (requires CREATE_NEW_PROCESS_GROUP at launch)
            try:
                p.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            except Exception:
                p.terminate()
        else:
            p.terminate()
        _wait_with_timeout(p, timeout)
    except Exception:
        pass
    finally:
        if p.poll() is None:
            # Still alive — force kill
            try:
                p.kill()
            except Exception:
                pass


def _wait_with_timeout(p: subprocess.Popen, timeout: float):
    # Poll until timeout to avoid waiting forever
    end = time.time() + timeout
    while time.time() < end:
        if p.poll() is not None:
            return
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description="Run backend and web services with supervision.")
    parser.add_argument("--backend", help="Backend script name or path (e.g., ai_service.py)")
    parser.add_argument("--web", help="Web script name or path (e.g., web_app.py)")
    args = parser.parse_args()

    backend_dir, backend_script, web_dir, web_script = _resolve_scripts(args.backend, args.web)

    print(f"[run] ROOT         = {ROOT}")
    print(f"[run] BACKEND_DIR  = {backend_dir}")
    print(f"[run] WEB_DIR      = {web_dir}")
    print(f"[run] BACKEND_CMD  = {sys.executable} {backend_script}")
    print(f"[run] WEB_CMD      = {sys.executable} {web_script}")

    p_backend = p_web = None
    exit_code = 0

    try:
        p_backend = launch("backend", backend_script, backend_dir)
        print(f"[run] Backend PID: {p_backend.pid}")

        # Stagger start slightly to reduce simultaneous stdout noise
        time.sleep(0.6)

        p_web = launch("web", web_script, web_dir)
        print(f"[run] Web PID    : {p_web.pid}")

        # Supervision loop: if either exits, bring down the other and exit
        while True:
            ret_b = p_backend.poll() if p_backend else 0
            ret_w = p_web.poll() if p_web else 0

            if ret_b is not None:
                print(f"[run] Backend exited with code {ret_b}. Shutting down web...")
                exit_code = ret_b
                break

            if ret_w is not None:
                print(f"[run] Web exited with code {ret_w}. Shutting down backend...")
                exit_code = ret_w
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[run] Ctrl+C received. Shutting down children...")
    except FileNotFoundError as e:
        print(f"[run] ERROR: {e}")
        exit_code = 2
    except Exception as e:
        print(f"[run] Unexpected error: {e}", file=sys.stderr)
        exit_code = 3
    finally:
        # Try graceful shutdowns
        if p_backend:
            graceful_terminate(p_backend, "backend")
        if p_web:
            graceful_terminate(p_web, "web")

    print(f"[run] Exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
