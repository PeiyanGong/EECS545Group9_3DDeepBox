import numpy

a = numpy.array([[1,2,3],[4,5,6]])
filename = "writetest.txt"
f = open(filename,"w")
for t in range(2):
    f.write(" ".join(map(str, a[t,:])))
    f.write("\n")
    # f.write("\n".join(" ".join(map(str, a[t,:]))))
    # for t in range(2):
# 	numpy.savetxt(filename, a[t,:], fmt="%d")