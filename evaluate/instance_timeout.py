"""Per-instance wall-clock limit (1200 s in the paper).

Each instance runs in its own test.py process, which is stopped when it exceeds
the limit: SIGTERM first, then SIGKILL after a grace period. Stopping a process
is the only reliable way to interrupt native code, such as a Julia search. An
instance that times out writes no result file and is listed in timeouts.txt
inside the output directory.
"""

import os
import signal
import subprocess
import sys
import time

KILL_AFTER = 30  # seconds between SIGTERM and SIGKILL


def _signal_process(process, sig):
    try:
        if os.name == 'posix':
            os.killpg(process.pid, sig)  # also stops the children of the process
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass


def run_with_timeout(cmd, timeout, kill_after=KILL_AFTER):
    """Run cmd; return (returncode, timed_out, elapsed_seconds)."""
    start = time.time()
    options = {'start_new_session': True} if os.name == 'posix' else {}
    process = subprocess.Popen(cmd, **options)
    try:
        returncode = process.wait(timeout=timeout)
        return returncode, False, time.time() - start
    except subprocess.TimeoutExpired:
        _signal_process(process, signal.SIGTERM)
    except BaseException:
        # e.g. Ctrl-C: do not leave the running instance behind.
        _signal_process(process, signal.SIGKILL)
        process.wait()
        raise

    try:
        returncode = process.wait(timeout=kill_after)
    except subprocess.TimeoutExpired:
        _signal_process(process, signal.SIGKILL)
        returncode = process.wait()
    return returncode, True, time.time() - start


def _record(path, key, status, elapsed):
    """Append one line per instance, so a repeated run adds no duplicates."""
    if os.path.exists(path):
        with open(path, encoding='utf-8') as handle:
            if any(line.startswith(key + '|') for line in handle):
                return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write('%s|%s|%ds\n' % (key, status, elapsed))


def run_instances(method_name, data_files, operators, args, seed, result_exists):
    """Run every instance of one method in its own process under args.timeout."""
    for index, test_file in enumerate(data_files, 1):
        if args.skip_existing and result_exists(args.output_dir, method_name, test_file, seed):
            print('  Skipping existing result: %s' % test_file)
            continue

        path = str(test_file).replace(os.sep, '/')
        relative = path[len('data/'):] if path.startswith('data/') else path
        cmd = [sys.executable, '-u', os.path.abspath(sys.argv[0]),
               '-m', method_name, '-d', relative,
               '-s', str(args.split), '--seed', str(seed),
               '-o', args.output_dir, '--timeout', '0']
        if operators:
            cmd += ['-p'] + list(operators)
        if args.skip_existing:
            cmd.append('--skip-existing')
        if not args.use_ground_truth:
            cmd.append('--no-use-ground-truth')

        print('  [%d/%d] %s' % (index, len(data_files), test_file))
        returncode, timed_out, elapsed = run_with_timeout(cmd, args.timeout)

        key = '%s|%s|seed%s' % (relative, method_name, seed)
        if timed_out:
            print('  Timed out after %ds: %s' % (elapsed, test_file))
            _record(os.path.join(args.output_dir, 'timeouts.txt'), key, 'timeout', elapsed)
        elif returncode != 0:
            print('  Failed with exit code %s: %s' % (returncode, test_file))
            _record(os.path.join(args.output_dir, 'failed.txt'), key, 'rc=%s' % returncode, elapsed)
