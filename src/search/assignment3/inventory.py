import hashlib
import getpass
import socket

Num_Workers = 3

MAX_PORT = 49152
MIN_PORT = 10000
BASE_PORT = int(hashlib.md5(getpass.getuser().encode()).hexdigest()[:8], 16) % \
       (MAX_PORT - MIN_PORT) + MIN_PORT + 5


port = BASE_PORT
URL = [ str(socket.gethostname()) + ":" + str(port + i) for i in range(Num_Workers)]

