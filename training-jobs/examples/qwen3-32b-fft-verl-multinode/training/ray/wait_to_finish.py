import time
import subprocess

while True:
    time.sleep(30)
    result = subprocess.run(["ray", "status"], capture_output=True, text=True)
    lines = result.stdout.split("\n")
    has_nodes = False
    for line in lines:
        if "node_" in line:
            has_nodes = True
    if not has_nodes:
        print(result.stdout)
        break

print("Detected no ray nodes; Exiting")