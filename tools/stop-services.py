import psutil

def main():
    with open("status.txt", "r") as f:
        line = f.readline().strip()
        pid = int(line)
        kill("service.py", pid)

def kill(procname, pid):
    for proc in psutil.process_iter():
        try:
            cmdline = proc.cmdline()
            if "python" in cmdline:
                print(f"Checking {cmdline} ...")
                if procname in cmdline: # Check if the script name is in the command line arguments
                    if proc.pid == pid:
                        proc.kill()
                        print(f"Killed process with PID: {proc.pid}")
                    else:
                        print(f"Process {procname} has pid {proc.pid} but given pid {pid} ... Not killing")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

if __name__ == '__main__':
    main()