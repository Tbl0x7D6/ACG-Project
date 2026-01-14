# Pos:
# 0.8987815380096436, -0.5891063213348389, 0.00018138557788915932, 

# Quat
# 0.6908290982246399, -0.15002840757369995, 0.3225850760936737, -0.6294326186180115

POS = (0.8987815380096436, -0.5891063213348389, 0.00018138557788915932)
QUAT = (0.69082910982246399, -0.15002840757369995, 0.3225850760936737, -0.6294326186180115)
TRANS1 = (127.75, -360, 400)
TRANS2 = (0, -256, 0)

# Quaternion helpers use layout (w, x, y, z)
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
    
def quat_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)
    
def rotate_vector(q, v):
    # Rotate a 3D vector v by quaternion q (w, x, y, z layout)
    q_v = (0.0, v[0], v[1], v[2])
    rotated = quat_mul(quat_mul(q, q_v), quat_conj(q))
    return rotated[1:]

def rotate_vector_inv(q, v):
    # Rotate vector v back by the inverse of quaternion q
    q_inv = quat_conj(q)
    q_v = (0.0, v[0], v[1], v[2])
    rotated = quat_mul(quat_mul(q_inv, q_v), q)
    return rotated[1:]

# Check again
import math
CENTER_OF_MASS = (0.037939, 0.065785, -0.038762)
CENTER_OF_MASS_SCALED = tuple(c * 256.0 for c in CENTER_OF_MASS)
HALF_ANGLE = 0.25 * math.pi  # 90° / 2
Q_X_90 = (math.cos(HALF_ANGLE), math.sin(HALF_ANGLE), 0.0, 0.0)
Q_Z_90 = (math.cos(HALF_ANGLE), 0.0, 0.0, math.sin(HALF_ANGLE))
Q_EXTRA = quat_mul(Q_Z_90, Q_X_90)  # Rz(90°) * Rx(90°) left-multiplied

delta_world = (
    (TRANS1[0] - TRANS2[0]) / 256.0,
    (TRANS1[1] - TRANS2[1]) / 256.0,
    (TRANS1[2] - TRANS2[2]) / 256.0,
)
pos_delta = rotate_vector_inv(Q_EXTRA, delta_world)
new_pos = (
    POS[0] + pos_delta[0],
    POS[1] + pos_delta[1],
    POS[2] + pos_delta[2],
)
print("New Position:", new_pos)

# New Position: (0.4787141084671021, 0.9940526485443115, 0.5012763405684386)

rotated_cm = rotate_vector(QUAT, CENTER_OF_MASS)
translated_pos = (
    POS[0] + rotated_cm[0],
    POS[1] + rotated_cm[1],
    POS[2] + rotated_cm[2],
)
final_pos = rotate_vector(Q_EXTRA, translated_pos)
final_pos_trans = (
    final_pos[0] * 256.0 + TRANS1[0],
    final_pos[1] * 256.0 + TRANS1[1],
    final_pos[2] * 256.0 + TRANS1[2],
)
print("Final Position with COM and extra rotation:", final_pos_trans)


        # pos_world = (
        #     data["position"][0] * 256.0,
        #     data["position"][1] * 256.0,
        #     data["position"][2] * 256.0,
        # )

        # rotated_cm = rotate_vector(q_state, CENTER_OF_MASS_SCALED)

        # # Equivalent single translation after T(cm) -> R(q) -> T(pos)
        # final_translation = (
        #     pos_world[0] + rotated_cm[0],
        #     pos_world[1] + rotated_cm[1],
        #     pos_world[2] + rotated_cm[2],
        # )
pos_world = (new_pos[0] * 256.0, new_pos[1] * 256.0, new_pos[2] * 256.0)
rotated_cm = rotate_vector(QUAT, CENTER_OF_MASS_SCALED)
final_translation = (
    pos_world[0] + rotated_cm[0],
    pos_world[1] + rotated_cm[1],
    pos_world[2] + rotated_cm[2],
)
final_pos = rotate_vector(Q_EXTRA, final_translation)
final_pos = (final_pos[0], final_pos[1] - 256, final_pos[2]) 
print("Final Position from Rigid States with COM and extra rotation:", final_pos)
print(
    "Difference (should be ~0):",
    (
        final_pos[0] - final_pos_trans[0],
        final_pos[1] - final_pos_trans[1],
        final_pos[2] - final_pos_trans[2],
    ),
)