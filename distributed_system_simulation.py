import random
import threading
import time
import queue

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.messages = queue.Queue()
        self.is_alive = True
        self.lock = threading.Lock()

    def send_message(self, message, recipient):
        if self.is_alive:
            recipient.receive_message(message, self.node_id)

    def receive_message(self, message, sender_id):
        self.messages.put((message, sender_id))

    def process_messages(self):
        while self.is_alive:
            try:
                message, sender_id = self.messages.get(timeout=1)
                print(f"Node {self.node_id} received message from Node {sender_id}: {message}")
                # Simulate processing time
                time.sleep(random.uniform(0.5, 1.5))
            except queue.Empty:
                continue

    def die(self):
        with self.lock:
            self.is_alive = False
            print(f"Node {self.node_id} has failed.")

class DistributedSystem:
    def __init__(self, num_nodes):
        self.nodes = [Node(i) for i in range(num_nodes)]
        self.threads = []

    def start(self):
        for node in self.nodes:
            thread = threading.Thread(target=node.process_messages)
            thread.start()
            self.threads.append(thread)

    def send_messages(self, from_node_id, message, to_node_id):
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        from_node.send_message(message, to_node)

    def simulate_failures(self, failure_rate=0.1):
        for node in self.nodes:
            if random.random() < failure_rate:
                node.die()

    def all_nodes_alive(self):
        return all(node.is_alive for node in self.nodes)

    def join(self):
        for thread in self.threads:
            thread.join()

def main():
    num_nodes = 5
    system = DistributedSystem(num_nodes)
    system.start()

    messages_to_send = [
        (0, "Hello from Node 0", 1),
        (1, "Hi from Node 1", 2),
        (0, "Node 0 to Node 2", 2),
        (3, "Node 3 here!", 0),
        (4, "Greetings from Node 4", 3),
    ]

    for from_node_id, message, to_node_id in messages_to_send:
        system.send_messages(from_node_id, message, to_node_id)
    
    # Simulate 1 second of sending messages and then check for failures.
    time.sleep(1)
    system.simulate_failures()

    # Check if all nodes are alive
    if not system.all_nodes_alive():
        print("Some nodes have failed.")

    # Give time for message processing before exiting
    time.sleep(5)
    system.join()
    print("Simulation complete.")

if __name__ == "__main__":
    main()