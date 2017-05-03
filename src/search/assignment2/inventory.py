import hashlib
import getpass


num_ind = 3
num_doc = 1
MAX_PORT = 49152
MIN_PORT = 10000
BASE_PORT = int(hashlib.md5(getpass.getuser().encode()).hexdigest()[:8], 16) % \
    (MAX_PORT - MIN_PORT) + MIN_PORT

#port1 = BASE_PORT + 1
port1 = BASE_PORT + 1
indUrl = [ str(port1 + i) + "/index?q=" for i in range(num_ind)]
		#	"localhost:" + str(port1 + 1) + "/index?q=",
		#	"localhost:" + str(port1 + 2) + "/index?q=" ]
			#"http://linserv2.cims.nyu.edu:" + str(port1 + 2) + "/index?q=" ]
#Iports = [35315,35316, 35317]
#indUrl=[]
#for port in Iports:
#	 indUrl.append("http://linserv2.cims.nyu.edu:" + str(port) + "/index?q=")

port2 = port1 + num_ind #35318
docUrl = [ str(port2 + i) + "/doc?id=" for i in range(num_doc)]

#Dports = [35318,35319]
#docUrl=[]
#for port in Dports:
#	 docUrl.append("http://linserv2.cims.nyu.edu:" + str(port) + "/doc?id=")