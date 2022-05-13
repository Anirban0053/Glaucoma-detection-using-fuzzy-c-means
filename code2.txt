import numpy as np
import cv2
import matplotlib.pyplot as plt


def init_mem_mat(nb_pixels, nb_clusters):
    mem_mat = np.zeros((nb_pixels, nb_clusters))
    x = np.arange(nb_pixels)
    for j in range(nb_clusters):
        xj = x % nb_clusters == j
        mem_mat[xj, j] = 1

    return mem_mat


def compute_centers(img_mat, mem_mat, fuzzy):
    num = np.dot(img_mat, mem_mat ** fuzzy)
    dem = np.sum(mem_mat ** fuzzy, axis=0)

    return num / dem


def update_mem_mat(ctr_mat, img_mat, fuzzy):
    ctr_mat_mesh, img_mat_mesh = np.meshgrid(ctr_mat, img_mat)
    power = 2. / (fuzzy - 1)
    p1 = abs(img_mat_mesh - ctr_mat_mesh) ** power
    p2 = np.sum((1. / abs(img_mat_mesh - ctr_mat_mesh)) ** power, axis=1)

    return 1. / (p1 * p2[:, None])


# Image to segment
src = cv2.imread(r"C:\Users\BUCKKA\Desktop\project\glau.jpg")
# src = cv2.imread("peppers.png")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
plt.show()

# Number of clusters
K = 10

# Number of data/pixels
N = gray.size

# Fuzzyness coefficient
m = 2

# Threshold
eps = 0.03

# Maximum number of iterations
max_i = 100
# Initialization
X = gray.flatten().astype('float')
U = init_mem_mat(N, K)

# Repeat until convergence
i = 0
while True:
    # Compute centroid for each cluster
    C = compute_centers(X, U, m)

    # Save initial membership matrix
    old_U = np.copy(U)

    # Update coefficients for each pixel
    U = update_mem_mat(C, X, m)

    # Difference between initial mem matrix and new one
    d = np.sum(abs(U - old_U))
    print(str(i) + " - d = " + str(d))

    # Check convergence
    if d < eps or i > max_i:
        break
    i += 1

# Segmentation
seg1 = np.argmax(U, axis=1)
seg = seg1.reshape(gray.shape).astype('int')
#cv2.imwrite(r"C:\Users\BUCKKA\Desktop\project\segmented image",seg)
# Plot
fig = plt.figure(figsize=(8,4),dpi=100)
xy=seg1[:]
d={}
for i in seg1:
    if(i not in d):
        d[i]=1
    else:
        d[i]+=1

for i in range(len(xy)):
    if(xy[i]==8 or xy[i]==9):
        continue
    else:
        xy[i]=0


xyz=xy.reshape(gray.shape).astype('int')
ax1 = fig.add_subplot(121)
ax1.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
ax1.set_title('original')

ax1 = fig.add_subplot(1, 2, 2)
ax1.imshow(xyz,cmap=None)
ax1.set_title('ROI')

# ax2 = fig.add_subplot(1,2,2)
# ax2.imshow(seg, cmap=None)
# ax2.set_title('segmentation')

plt.show()
m=list(seg1)
cup=m.count(8)
outer=m.count(9)
disc=cup+outer 
print(cup/disc)