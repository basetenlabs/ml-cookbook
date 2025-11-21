"""
rendezvous ensures that all nodes are ready to start training
"""

import subprocess
import os

timeout_mins = 5
import time 

start = time.time()
print("Rendezvous started at {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
print("Rendezvous will timeout in {} minutes".format(timeout_mins))

num_nodes = int(os.environ["BT_GROUP_SIZE"])
while True:
    nodes_up = 0
    result = subprocess.run(["ray", "status"], capture_output=True, text=True)
    lines = result.stdout.split("\n")
    for line in lines:
        if "node_" in line:
            nodes_up += 1
    if nodes_up == num_nodes:
        break
    if time.time() - start > timeout_mins * 60:
        print("Most Recent Ray Status:")
        print(result.stdout)
        raise RuntimeError("Rendezvous timed out; Only {} nodes are up".format(nodes_up))
    time.sleep(10)

print("Rendezvous complete; All nodes are up")