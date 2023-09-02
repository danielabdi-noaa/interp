import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", "-i", dest="input", required=True, help="Path to input text file"
)
parser.add_argument(
    "--output", "-o", dest="output", required=True, help="Path to output png file"
)
args = parser.parse_args()

# Read input data from text file
x, y, z = [], [], []
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
            z.append(vals[2:])
x = np.array(x)
y = np.array(y)
z = np.array(z)

# Plot all fields
for i in range(np.size(z,1)):

    filt = (z[:,i] != 9999.00)
    xi = x[filt]
    yi = y[filt]
    zi = z[filt,i]

    fig, axs = plt.subplots(1, 1, figsize=[13.63, 10.0])
    pc = axs.tricontourf(xi, yi, zi, cmap='viridis')
    #pc = axs.scatter(xi, yi, c=zi, s=0.03, cmap="viridis")
    fig.colorbar(pc, ax=axs, extend='both')
    plt.savefig(f"field_{i}_{args.output}")
