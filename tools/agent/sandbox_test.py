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
        env = dict(os.environ, KIDI_AGENT_HOME=str(self.root / "runtime"), KIDI_AGENT_CACHE=str(self.root / "cache"))
        self.env = env

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_sandboxed(self, command, *options, env=None):
        return subprocess.run([str(HERE / "sandbox.sh"), *options, "--exec", str(self.ws), "--", "bash", "-c", command],
                              env=env or self.env, capture_output=True, text=True)

    def test_agent_provider_uses_configured_endpoint_and_model_id(self):
        env = dict(self.env, KIDI_AGENT_BASE_URL="http://127.0.0.1:8123/v1", KIDI_AGENT_MODEL_ID="test-model")
        command = (
            "python3 -c 'import json, os; c=json.load(open(os.path.join(os.environ[\"PI_CODING_AGENT_DIR\"], "
            "\"models.json\"))); p=c[\"providers\"][\"kidi\"]; "
            "assert p[\"baseUrl\"] == \"http://127.0.0.1:8123/v1\"; "
            "assert p[\"models\"][0][\"id\"] == \"test-model\"'"
        )
        result = self.run_sandboxed(command, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)

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


@unittest.skipUnless(HAS_LANDLOCK, "requires Linux with Landlock")
class KidiCodeLauncherTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.ws = self.root / "ws"
        self.ws.mkdir()
        self.rt = self.root / "runtime"
        self.cache = self.root / "cache"
        pi_bin = self.rt / "pi" / "node_modules" / ".bin" / "pi"
        pi_bin.parent.mkdir(parents=True, exist_ok=True)
        pi_bin.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                          "cfg = json.load(open(os.path.join(os.environ['PI_CODING_AGENT_DIR'], 'models.json')))\n"
                          "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'models': cfg}))\n")
        pi_bin.chmod(0o755)
        node_bin = self.rt / "node" / "bin"
        node_bin.mkdir(parents=True, exist_ok=True)
        self.env = dict(
            os.environ,
            KIDI_AGENT_HOME=str(self.rt),
            KIDI_AGENT_CACHE=str(self.cache),
            KIDI_HUB=str(self.root / "hub"),
        )

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_reuses_existing_server_and_forwards_pi_args(self):
        import http.server
        import json
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"object": "list", "data": [{"id": "served-test-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            res = subprocess.run(
                [sys.executable, str(HERE / "kidi-code"), "-w", str(self.ws), "--port", str(port),
                 "--context-size", "16384", "--max-new-tokens", "2048", "-p", "hello"],
                env=self.env, capture_output=True, text=True, check=True,
            )
        finally:
            srv.shutdown()
            srv.server_close()

        out = json.loads(res.stdout)
        self.assertEqual(out["cwd"], str(self.ws))
        self.assertIn("-p", out["argv"])
        self.assertIn("hello", out["argv"])
        model = out["models"]["providers"]["kidi"]["models"][0]
        self.assertEqual(model["id"], "served-test-model")
        self.assertEqual(model["contextWindow"], 16384)
        self.assertEqual(model["maxTokens"], 2048)

    def test_autostarts_and_stops_kidi_serve_from_hub_cache(self):
        import json

        model_dir = self.root / "hub" / "thinkingmachines" / "Inkling-2-Mini"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        pid_file = self.root / "server.pid"
        args_file = self.root / "server.args"
        mock_kidi = self.root / "mock_kidi"
        mock_kidi.write_text(
            "#!/usr/bin/env python3\n"
            "import http.server, json, os, signal, sys\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            f"open({str(args_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
            "class H(http.server.BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        b = b'{\"data\":[{\"id\":\"kidi-local\"}]}'\n"
            "        self.send_response(200); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)\n"
            "    def log_message(self, *a): pass\n"
            "srv = http.server.HTTPServer(('127.0.0.1', port), H)\n"
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
            "srv.serve_forever()\n"
        )
        mock_kidi.chmod(0o755)

        res = subprocess.run(
            [sys.executable, str(HERE / "kidi-code"), "--kidi-bin", str(mock_kidi),
             "--backend", "auto", "-w", str(self.ws), "--port", str(port), "-p", "ping"],
            env=self.env, capture_output=True, text=True, check=True,
        )
        out = json.loads(res.stdout)
        self.assertEqual(out["models"]["providers"]["kidi"]["models"][0]["id"], "kidi-local")
        srv_args = json.loads(args_file.read_text())
        self.assertEqual(srv_args[0], "serve")
        self.assertIn(str(model_dir), srv_args)
        srv_pid = int(pid_file.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(srv_pid, 0)


if __name__ == "__main__":
    unittest.main()
