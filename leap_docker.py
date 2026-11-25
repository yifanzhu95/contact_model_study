#!/usr/bin/env python3
"""Run Docker using an existing docker-group membership, including in older shells."""
import grp
import os
import pwd
import shlex
import shutil
import sys


def docker_command(arguments):
    executable = shutil.which("docker")
    if executable is None:
        raise RuntimeError("Docker executable was not found in PATH")
    command = [executable, *arguments]
    try:
        group = grp.getgrnam("docker")
        username = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return command
    active_groups = {os.getgid(), *os.getgroups()}
    if group.gr_gid not in active_groups and username in group.gr_mem:
        sg = shutil.which("sg")
        if sg is None:
            raise RuntimeError("Docker membership is inactive in this shell and sg is unavailable; log in again")
        # sg reuses membership already granted by the administrator. Quote
        # every argument because sg accepts a shell command, not an argv array.
        return [sg, "docker", "-c", shlex.join(command)]
    return command


def main(argv=None):
    command = docker_command(sys.argv[1:] if argv is None else argv)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
