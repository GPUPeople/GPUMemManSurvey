import subprocess
import threading
import signal
import os

""" Run system commands with timeout
"""
class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.out = None

    def run_command(self, capture = False, working_directory = ""):
        if not capture:
            self.process = subprocess.Popen(self.cmd,shell=True)
            self.process.communicate()
            return
        # capturing the outputs of shell commands
        self.process = subprocess.Popen(self.cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE, cwd=working_directory)
        out,err = self.process.communicate()
        if len(out) > 0:
            self.out = out.splitlines()
        else:
            self.out = None

    # set default timeout to 10 minutes
    def run(self, capture = False, timeout = 600, working_directory = ""):
        thread = threading.Thread(target=self.run_command, args=(capture, working_directory))
        thread.start()
        thread.join(timeout)
        thread_killed = False
        if thread.is_alive():
            print('Command timeout, kill it: ' + self.cmd)
            if os.name == 'nt': # If on Windows
                os.system("TASKKILL /F /T /PID {0}".format(self.process.pid))
                # os.system("TASKKILL /F /PID {0}".format(self.process.pid))
                # os.kill(self.process.pid, signal.CTRL_C_EVENT)
                # os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                # self.process.terminate()
            else:
                self.process.terminate()
            thread.join()
            thread_killed = True
        return self.out, thread_killed


# Example

# from timedprocess import Command

# Command('pwd').run() # Without timeout
# Command('echo "sleep 10 seconds"; sleep 10; echo "done"').run(timeout=2) # With timeout