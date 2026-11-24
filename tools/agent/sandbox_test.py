"""Linux-only checks for the agent sandbox: python3 -m unittest tools/agent/sandbox_test.py"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import landlock_run  # noqa: E402

SYSTEM_RO = ["/usr", "/bin", "/lib", "/lib64", "/etc", "/proc", "/sys"]
HAS_LANDLOCK = sys.platform == "linux" and landlock_run.landlock_abi() >= 1
HAS_SUDO = HAS_LANDLOCK and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                   capture_output=True)


@unittest.skipUnless(HAS_LANDLOCK, "requires Linux with Landlock")
class LandlockRunTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.ws = self.root / "ws"
        self.ws.mkdir()
        (self.root / "secret").write_text("secret")

    def tearDown(self):
        shutil.rmtree(self.root)

    def confined(self, script, *extra):
        args = [sys.executable, str(HERE / "landlock_run.py"), "--rw", str(self.ws), "--rw", "/dev/null"]
        for path in SYSTEM_RO:
            args += ["--ro", path]
        return subprocess.run([*args, *extra, "--", "/usr/bin/python3", "-c", script],
                              cwd=self.ws, capture_output=True, text=True)

    def test_only_listed_paths_are_accessible(self):
        (self.ws / "link").symlink_to(self.root / "secret")
        script = f"""
import os
open("inside", "w").write("ok")
for action in (lambda: open({str(self.root / 'secret')!r}).read(), lambda: open("link").read(),
               lambda: open({str(self.root / 'new')!r}, "w"), lambda: os.rename("inside", {str(self.root / 'moved')!r})):
    try:
        action()
        print("escaped")
    except PermissionError:
        pass
"""
        result = self.confined(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual((self.ws / "inside").read_text(), "ok")

    @unittest.skipUnless(HAS_LANDLOCK and landlock_run.landlock_abi() >= 4, "requires Landlock ABI 4")
    def test_tcp_connect_limited_to_listed_ports(self):
        allowed, denied = socket.create_server(("127.0.0.1", 0)), socket.create_server(("127.0.0.1", 0))
        with allowed, denied:
            script = f"""
import socket
socket.create_connection(("127.0.0.1", {allowed.getsockname()[1]})).close()
for action in (lambda: socket.create_connection(("127.0.0.1", {denied.getsockname()[1]})),
               lambda: socket.create_server(("127.0.0.1", 0))):
    try:
        action()
        print("escaped")
    except PermissionError:
        pass
"""
            result = self.confined(script, "--tcp-connect", str(allowed.getsockname()[1]))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


@unittest.skipUnless(HAS_LANDLOCK, "requires Linux with Landlock")
class SandboxScriptTest(unittest.TestCase):
    """A linked worktree, the layout used for agent workspaces."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        main = self.root / "main"
        main.mkdir()
        git("init", "-q", cwd=main)
        (main / "a.py").write_text("x = 1\n")
        git("add", "a.py", cwd=main)
        git("commit", "-qm", "init", cwd=main)
        git("worktree", "add", "-q", str(self.root / "ws"), cwd=main)
        self.ws = self.root / "ws"
        self.hooks = main / ".git" / "hooks"
        env = dict(os.environ, KIDI_AGENT_CACHE=str(self.root / "cache"))
        self.env = env

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_sandboxed(self, command, *options, env=None):
        return subprocess.run([str(HERE / "sandbox.sh"), *options, str(self.ws), "--", "bash", "-c", command],
                              env=env or self.env, capture_output=True, text=True)

    def test_model_endpoint_is_passed_through_and_reachable_only_on_its_port(self):
        env = dict(self.env, KIDI_AGENT_BASE_URL="http://127.0.0.1:8123/v1")
        result = self.run_sandboxed('test "$KIDI_AGENT_BASE_URL" = http://127.0.0.1:8123/v1', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tcp 8123", result.stderr)

    def test_python_venv_and_git_review_workflow(self):
        result = self.run_sandboxed(
            "python -c 'import sys' && python3 -m venv .venv && .venv/bin/python -c 'import a' && "
            "echo 'x = 2' > a.py && git diff --name-only | grep -qx a.py && "
            "! git add a.py 2>/dev/null && ! touch " + str(self.hooks / "pre-commit"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.hooks / "pre-commit").exists())
        auto = self.run_sandboxed('test "$VIRTUAL_ENV" = "$PWD/.venv" && test "$(command -v python)" = "$PWD/.venv/bin/python"')
        self.assertEqual(auto.returncode, 0, auto.stderr)

    @unittest.skipUnless(HAS_SUDO, "requires passwordless sudo")
    def test_mount_namespace_protects_gitfile_and_provides_shm(self):
        gitfile = (self.ws / ".git").read_text()
        result = self.run_sandboxed(
            "python3 -c 'import multiprocessing as m; m.Lock()' && test -z \"$(ls /dev/shm)\" && "
            "! (echo 'gitdir: /tmp' > .git) 2>/dev/null")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.ws / ".git").read_text(), gitfile)


if __name__ == "__main__":
    unittest.main()
