import hashlib
import getpass


num_ind = 3
num_doc = 1
MAX_PORT = 49152
MIN_PORT = 10000
BASE_PORT = int(hashlib.md5(getpass.getuser().encode()).hexdigest()[:8], 16) % \
    (MAX_PORT - MIN_PORT) + MIN_PORT

port1 = BASE_PORT + 1
indUrl = [ str(port1 + i) + "/index?q=" for i in range(num_ind)]

port2 = port1 + num_ind #35318
docUrl = [ str(port2 + i) + "/docA?id=" for i in range(num_doc)]
