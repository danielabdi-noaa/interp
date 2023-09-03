import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import struct

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", "-i", dest="input", required=True, help="Path to input text file"
)
parser.add_argument(
    "--output", "-o", dest="output", required=True, help="Path to output png file"
)
args = parser.parse_args()

is3d = False

# Read input data from text file
x, y, z, v = [], [], [], []

if "txt" in args.input:
    with open(args.input, "r") as file:
        first = True
        for line in file:
            if first:
                first = False
                pass
            else:
                vals = list(map(float, line.split()))
                x.append(vals[0])
                y.append(vals[1])
                if is3d:
                    z.append(vals[2])
                    v.append(vals[3:])
                else:
                    v.append(vals[2:])
else:
    with open(args.input, 'rb') as file:
        numPoints, numFields = struct.unpack('ii', file.read(8))
        for i in range(numPoints):
            buffer = file.read((2+numFields)*8)
            if is3d:
                xi, yi, zi, *vi = struct.unpack('ddd' + 'd'*numFields, buffer)
            else:
                xi, yi, *vi = struct.unpack('dd' + 'd'*numFields, buffer)
            x.append(xi)
            y.append(yi)
            if is3d:
                z.append(zi)
                v.append(vi)
            else:
                v.append(vi)

x = np.array(x)
y = np.array(y)
if is3d:
    z = np.array(z)
    v = np.array(v)
else:
    v = np.array(v)

# Plot all fields
for i in range(np.size(v,1)):

    filt = (v[:,i] != 9999.00)
    xi = x[filt]
    yi = y[filt]
    if is3d:
        zi = z[filt]
    vi = v[filt,i]

    if is3d:
        fig = plt.figure()
        axs = fig.add_subplot(111, projection='3d')
        #pc = axs.plot_trisurf(xi, yi, zi, cmap='viridis')
        pc = axs.scatter(xi, yi, zi, c=vi, cmap='viridis', s= 0.03)
    else:
        fig, axs = plt.subplots(1, 1, figsize=[13.63, 10.0])
        pc = axs.tricontourf(xi, yi, vi, cmap='viridis')
        #pc = axs.scatter(xi, yi, c=vi, s=0.03, cmap="viridis")
    fig.colorbar(pc, ax=axs, extend='both')
    plt.savefig(f"field_{i}_{args.output}")
    #plt.show()
