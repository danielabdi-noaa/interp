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
    for line in file:
        vals = list(map(float, line.split()))
        x.append(vals[0])
        y.append(vals[1])
        z.append(vals[2])
x = np.array(x)
y = np.array(y)
z = np.array(z)

fig, axs = plt.subplots(1, 1, figsize=[13.63, 10.0])
# axs.tricontourf(x, y, z, cmap='viridis')
axs.scatter(x, y, c=z, s=0.03, cmap="viridis")

plt.savefig(args.output)
