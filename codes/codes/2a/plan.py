import matplotlib.pyplot as plt
import math

# List of [x, y, yaw]
points = [
    [0.11, -5.54, 0.20],
    [0.26, -1.95, 1.57],
    [0.2, 0.2, 1.89],
    [-0.35, 1.19, 2.80],
    [-0.77, 1.15, 3.12],
    [-0.75, 1.01, -3.06],
    [-0.91, 1.02, -2.80],
    [-1.48, 0.07, -1.57],
    [-1.53, -6.61, -1.57]
]

# Extract x, y, theta
xs = [p[0] for p in points]
ys = [p[1] for p in points]
thetas = [p[2] for p in points]

plt.figure(figsize=(7, 7))

# Plot points
plt.scatter(xs, ys, color='blue')

# Draw orientation arrows
for x, y, theta in points:
    dx = math.cos(theta) * 0.2   # arrow length scaling
    dy = math.sin(theta) * 0.2
    plt.arrow(x, y, dx, dy, head_width=0.05, color='red')

plt.title("Robot Pose Plot (x, y, orientation)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')

plt.show()
