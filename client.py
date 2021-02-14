import socket
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to image that should be predicted by the model')
args = parser.parse_args()
img_path = args.path
server_address = ('0.0.0.0', 4224)

s = socket.socket()
s.connect(server_address)

s.send(img_path.encode())
while 1:
    data = s.recv(4096)
    if data:
        print(data)
        break

s.send('exit'.encode())
s.close()
