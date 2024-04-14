import socket
import threading

class AutoChatRoom:
    def __init__(self):
        self.server_socket = None
        self.clients = {}
        self.admin_password = "admin123"
        self.banned_addresses = []

    def create_server(self, server_name):
        self.server_socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        self.server_socket.bind((server_name, 1))
        self.server_socket.listen(5)
        print(f"Server started with name: {server_name}")

        while True:
            client_socket, address = self.server_socket.accept()
            print(f"New client connected: {address}")
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, address))
            client_thread.start()

    def handle_client(self, client_socket, address):
        if address[0] in self.banned_addresses:
            client_socket.send("You are banned from this chat room.".encode())
            client_socket.close()
            return

        client_socket.send("Enter your username: ".encode())
        username = client_socket.recv(1024).decode().strip()
        self.clients[client_socket] = username

        while True:
            message = client_socket.recv(1024).decode().strip()

            if message.startswith("/admin"):
                password = message.split(" ")[1]
                if password == self.admin_password:
                    client_socket.send("Admin access granted.".encode())
                else:
                    client_socket.send("Invalid admin password.".encode())
            elif message.startswith("/kick"):
                if self.clients[client_socket] == "admin":
                    target_username = message.split(" ")[1]
                    for sock, name in self.clients.items():
                        if name == target_username:
                            sock.send("You have been kicked from the chat room.".encode())
                            sock.close()
                            del self.clients[sock]
                            break
                else:
                    client_socket.send("You don't have permission to kick users.".encode())
            elif message.startswith("/ban"):
                if self.clients[client_socket] == "admin":
                    target_username = message.split(" ")[1]
                    for sock, name in self.clients.items():
                        if name == target_username:
                            self.banned_addresses.append(sock.getpeername()[0])
                            sock.send("You have been banned from the chat room.".encode())
                            sock.close()
                            del self.clients[sock]
                            break
                else:
                    client_socket.send("You don't have permission to ban users.".encode())
            else:
                self.broadcast(f"{self.clients[client_socket]}: {message}", client_socket)

    def broadcast(self, message, sender_socket):
        for client_socket in self.clients:
            if client_socket != sender_socket:
                client_socket.send(message.encode())

    def create_client(self, server_name):
        client_socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        client_socket.connect((server_name, 1))

        username = input(client_socket.recv(1024).decode())
        client_socket.send(username.encode())

        receiving_thread = threading.Thread(target=self.receive_messages, args=(client_socket,))
        receiving_thread.start()

        while True:
            message = input()
            client_socket.send(message.encode())

    def receive_messages(self, client_socket):
        while True:
            try:
                message = client_socket.recv(1024).decode()
                print(message)
            except:
                break