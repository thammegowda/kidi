#!/usr/bin/env python3
"""Run a command confined by Landlock (Linux >= 5.13, no privileges needed).

  landlock_run.py [--rw PATH]... [--ro PATH]... [--tcp-connect PORT]... -- CMD [ARGS...]

Only listed paths are accessible; --ro paths allow read and execute. The restriction is
inherited by every child process and cannot be lifted. On Landlock ABI >= 4, TCP bind is
denied and TCP connect is limited to --tcp-connect ports. Fails closed if Landlock is
unavailable.
"""
import argparse
import ctypes
import os
import struct
import sys

SYS_CREATE_RULESET, SYS_ADD_RULE, SYS_RESTRICT_SELF = 444, 445, 446
CREATE_RULESET_VERSION = 1
RULE_PATH_BENEATH, RULE_NET_PORT = 1, 2
PR_SET_NO_NEW_PRIVS = 38

EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REFER, TRUNCATE = 1 << 13, 1 << 14
FILE_RIGHTS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE
READ_RIGHTS = EXECUTE | READ_FILE | READ_DIR
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1

libc = ctypes.CDLL(None, use_errno=True)


def check(result):
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def landlock_abi():
    try:
        return check(libc.syscall(SYS_CREATE_RULESET, None, 0, CREATE_RULESET_VERSION))
    except OSError:
        return 0


def fs_rights(abi):
    # Device ioctls (ABI 5) are left unhandled so terminals keep working.
    rights = (1 << 13) - 1
    if abi >= 2:
        rights |= REFER
    if abi >= 3:
        rights |= TRUNCATE
    return rights


def add_path_rule(ruleset, path, access):
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        if not os.path.isdir(path):
            access &= FILE_RIGHTS
        attr = ctypes.create_string_buffer(struct.pack("=Qi", access, fd))
        check(libc.syscall(SYS_ADD_RULE, ruleset, RULE_PATH_BENEATH, attr, 0))
    finally:
        os.close(fd)


def restrict(rw_paths, ro_paths, tcp_ports):
    """Confine the current process; returns the Landlock ABI version used."""
    abi = landlock_abi()
    if abi < 1:
        raise OSError("Landlock is not available on this kernel")
    all_fs = fs_rights(abi)
    handled_net = NET_BIND_TCP | NET_CONNECT_TCP if abi >= 4 else 0
    attr = ctypes.create_string_buffer(struct.pack("=QQ", all_fs, handled_net))
    ruleset = check(libc.syscall(SYS_CREATE_RULESET, attr, 16 if abi >= 4 else 8, 0))
    try:
        for path in ro_paths:
            if os.path.exists(path):
                add_path_rule(ruleset, path, READ_RIGHTS & all_fs)
        for path in rw_paths:
            add_path_rule(ruleset, path, all_fs)
        if abi >= 4:
            for port in tcp_ports:
                rule = ctypes.create_string_buffer(struct.pack("=QQ", NET_CONNECT_TCP, port))
                check(libc.syscall(SYS_ADD_RULE, ruleset, RULE_NET_PORT, rule, 0))
        check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))
        check(libc.syscall(SYS_RESTRICT_SELF, ruleset, 0))
    finally:
        os.close(ruleset)
    return abi


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rw", action="append", default=[], help="read/write path (must exist)")
    parser.add_argument("--ro", action="append", default=[], help="read/execute path (skipped if missing)")
    parser.add_argument("--tcp-connect", action="append", type=int, default=[], help="allowed TCP connect port")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command")
    try:
        abi = restrict(args.rw, args.ro, args.tcp_connect)
    except OSError as error:
        sys.exit(f"landlock_run: {error}")
    if abi < 4:
        print(f"landlock_run: warning: Landlock ABI {abi} cannot restrict TCP", file=sys.stderr)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
