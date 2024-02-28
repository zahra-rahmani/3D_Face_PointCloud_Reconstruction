import mediapipe as mp
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from scipy.interpolate import griddata



font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (255, 255, 255) 

Nta = 25
selectedpoints = [0, 2, 4, 8, 17, 33, 40, 55, 70, 98, 133, 159,
                  263, 276, 285, 287, 327, 362, 386]
K = np.array([[1.90327008e+03, 0.00000000e+00, 3.66893302e+02],
       [0.00000000e+00, 1.85168320e+03, 3.04712415e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
def distance(a, b):
    return ((a[0] - b[0])** 2 + (a[1] - b[1])** 2)**(1/2)
def select_n_points(points, treshold):
    sp = []
    for j in range(len(points)):
        point = points[j]
        f = 1
        for i in range(len(sp)):
            if distance(point, points[sp[i]]) < treshold:
                f = 0
                break
        if f == 1:
            sp.append(j)
    return sp

def compute_essential_matrix(pair_points1, pair_points2, K1, K2):
    assert len(pair_points1) == len(pair_points2), "Number of point pairs should be the same"

    # Convert point pairs to homogeneous coordinates
    points1 = np.array(pair_points1 + [[1]] * len(pair_points1))
    points2 = np.array(pair_points2 + [[1]] * len(pair_points2))

    # Normalize the coordinates
    points1_normalized = cv2.undistortPoints(points1.reshape(-1, 1, 2), K1, None)
    points2_normalized = cv2.undistortPoints(points2.reshape(-1, 1, 2), K2, None)

    # Compute the essential matrix
    E, _ = cv2.findEssentialMat(points1_normalized, points2_normalized, focal=1.0, pp=(0, 0))

    return E

def compute_essential_matrix_8point(points1, points2, K1, K2):
    assert len(points1) == len(points2) >= 8, "At least 8 corresponding points are required"

    # Normalize the coordinates
    points1_normalized = cv2.undistortPoints(np.array(points1).reshape(-1, 1, 2), K1, None)
    points2_normalized = cv2.undistortPoints(np.array(points2).reshape(-1, 1, 2), K2, None)

    # Create the A matrix for the linear equation (F = [a, b, c; d, e, f; g, h, i])
    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        u1, v1 = points1_normalized[i].squeeze()
        u2, v2 = points2_normalized[i].squeeze()
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]

    # Solve for the singular value decomposition of A
    _, _, V = np.linalg.svd(A)

    # Extract the smallest singular value corresponding to the essential matrix
    F = V[-1].reshape(3, 3)

    # Enforce the rank-2 constraint on F
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    # Denormalize the essential matrix
    E = np.dot(K2.T, np.dot(F, K1))

    return E


# Initialize the face mesh solution
mp_face_mesh = mp.solutions.face_mesh

# Create a face mesh instance
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

c = []
for i in range(1000):
    c.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

Xr_baseline = []
Yr_baseline = []
Zr_baseline = []

Xl_baseline = []
Yl_baseline = []
Zl_baseline = []

r_camera = cv2.imread('IMG_6291.png')
l_camera = cv2.imread('IMG_6290.png')
# Convert the frame to RGB
rgb_r_camera = cv2.cvtColor(r_camera, cv2.COLOR_BGR2RGB)
rgb_l_camera = cv2.cvtColor(l_camera, cv2.COLOR_BGR2RGB)

# Preprocess the frame for face detection
r_results = face_mesh.process(rgb_r_camera)
rc_points = []
for landmark in r_results.multi_face_landmarks[0].landmark:
    rc_points.append([landmark.x * r_camera.shape[1],
                      landmark.y * r_camera.shape[0]])
    Xr_baseline.append(landmark.x * r_camera.shape[1])
    Yr_baseline.append(landmark.y * r_camera.shape[0])
    Zr_baseline.append(landmark.z * 50)
sp = select_n_points(rc_points, 20)
sp = selectedpoints

# Check if any faces were detected
if r_results.multi_face_landmarks:
    # Get the landmarks for the first detected face
    face_landmarks = r_results.multi_face_landmarks[0]

    # Draw the landmarks on the frame
    for i, landmark in enumerate(face_landmarks.landmark):
        x = int(landmark.x * r_camera.shape[1])
        y = int(landmark.y * r_camera.shape[0])
        if i in sp:
            cv2.circle(r_camera, (x, y), 5, (c[i][0], c[i][1], c[i][2]), -1)
            cv2.putText(r_camera, str(i), (x, y), font, font_scale, font_color, font_thickness)
        

# Display the frame with detected facial landmarks
output = cv2.resize(r_camera, (int(r_camera.shape[1] * 0.5), int(l_camera.shape[0] * 0.5)))
cv2.imshow('R', output)

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
l_results = face_mesh.process(rgb_l_camera)

# Check if any faces were detected
if l_results.multi_face_landmarks:
    # Get the landmarks for the first detected face
    face_landmarks = l_results.multi_face_landmarks[0]

    # Draw the landmarks on the frame
    for i, landmark in enumerate(face_landmarks.landmark):
        x = int(landmark.x * l_camera.shape[1])
        y = int(landmark.y * l_camera.shape[0])
        #cv2.circle(l_camera, (x, y), 5, (c[i][0], c[i][1], c[i][2]), -1)
        if i in sp:
            cv2.circle(l_camera, (x, y), 5, (c[i][0], c[i][1], c[i][2]), -1)


 
# Display the frame with detected facial landmarks
output = cv2.resize(l_camera, (int(l_camera.shape[1] * 0.5), int(l_camera.shape[0] * 0.5)))
cv2.imshow('L', output)
cv2.waitKey(1000)


rc_points = []
for landmark in r_results.multi_face_landmarks[0].landmark:
    rc_points.append([landmark.x * r_camera.shape[1],
                      landmark.y * r_camera.shape[0]])

lc_points = []
for landmark in l_results.multi_face_landmarks[0].landmark:
    lc_points.append([landmark.x * l_camera.shape[1],
                      landmark.y * l_camera.shape[0]])


indexes = []
apd_R = []
apd_L = []
N = 100
#sp = select_n_points(rc_points, 10)
for i in range(0, len(sp)):
    #indexes.append(random.randint(0, 467))
    indexes.append(i)
    apd_R.append(rc_points[sp[i]])
    apd_L.append(lc_points[sp[i]])

E = compute_essential_matrix(np.array(apd_R), np.array(apd_L), K, K)

#E = compute_essential_matrix(rc_points, lc_points)
E = np.array([[0, 0, 0],
              [0, 0, -20],
              [0, 20, 0]])
def decompose_essential_matrix(E):
    """
    Decompose the essential matrix into possible rotation and translation matrices.

    Parameters:
    - E: Essential matrix (3x3 matrix).

    Returns:
    - Rotation matrices (R1, R2) and translation vector (t).
    """
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

    # Two possible rotations
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))

    # Ensure proper rotation matrix
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Translation
    t = U[:, 2]

    return R1, R2, t


def triangulate_points(P1, P2, pts1, pts2):

    print(pts1.shape, pts2.shape)
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D / pts4D[3]
    return pts3D[:3]



# Find rotation and translation from the essential matrix
R1, R2, t = decompose_essential_matrix(E)

# Choose the correct R and t here (there are 4 possibilities)
# This often requires additional constraints or information

# Create projection matrices for the two cameras
P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(K, np.hstack((np.eye(3), t.reshape(3, 1))))  # Replace R1 and t with chosen values

# Convert points to homogeneous coordinates
pts1_homo = cv2.convertPointsToHomogeneous(np.array(rc_points))[:, 0, :2] # Extract x, y and transpose
pts2_homo = cv2.convertPointsToHomogeneous(np.array(lc_points))[:, 0, :2]


# Triangulate points
points_3D = triangulate_points(P1, P2, pts1_homo, pts2_homo)

X = points_3D[0]
Y = points_3D[1]
Z = points_3D[2]

new_X = []
new_Y = []
new_Z = []

"""for i in range(len(X)):
    if -10 < X[i] < 10 and -10 < Y[i] < 10 and -10 < Z[i] < 10:
        new_X.append(X[i])
        new_Y.append(Y[i])
        new_Z.append(Z[i])
X = np.array(new_X)
Y = np.array(new_Y)
Z = np.array(new_Z)"""

# Creating a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3D points
ax.scatter(X, Y, Z)
#ax.scatter(X*200+600, Y*200+600, Z*4 - 48, color="r")
#ax.scatter(Xr_baseline, Yr_baseline, Zr_baseline, color = 'b')

# Setting labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Setting title
ax.set_title('3D Facial Mask Points Visualization')

# Show plot
plt.show()

print(R1, R2)


# ______________________________________


# Create grid values first.
xi = np.linspace(min(X), max(X), 100)
yi = np.linspace(min(Y), max(Y), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate; there's also method='cubic' for 2-D data such as here
zi = griddata((X, Y), Z, (xi, yi), method='linear')

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotting the triangulated surface with simulated lighting
surf = ax.plot_surface(xi, yi, zi, cmap='gist_gray', edgecolor='none', shade=True)

# Setting labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Setting title
ax.set_title('3D Triangulated Facial Surface with Simulated Lighting')

# Show plot
plt.show()