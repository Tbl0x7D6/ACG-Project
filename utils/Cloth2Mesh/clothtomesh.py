import taichi as ti
from pathlib import Path
import math

# Center of mass in simulation space (same frame as rigid_body.py)
CENTER_OF_MASS = (0.037939, 0.065785, -0.038762)
SCALE = 0.3
TRANSLATION_OFFSET = (127.75, -360, 400)

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


HALF_ANGLE = 0.25 * math.pi  # 90° / 2
Q_X_90 = (math.cos(HALF_ANGLE), math.sin(HALF_ANGLE), 0.0, 0.0)
Q_Z_90 = (math.cos(HALF_ANGLE), 0.0, 0.0, math.sin(HALF_ANGLE))
Q_EXTRA = quat_mul(Q_Z_90, Q_X_90)

def write_ply(path: Path, x: ti.template(), n: int):
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n * n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {(n - 1) * (n - 1) * 2}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i in range(n):
            for j in range(n):
                f.write(f"{x[i, j].z * 256.0 + TRANSLATION_OFFSET[0]} {x[i, j].x * 256.0 + TRANSLATION_OFFSET[1]} {x[i, j].y * 256.0 + TRANSLATION_OFFSET[2]}\n")

        for i in range(n - 1):
            for j in range(n - 1):
                f.write(f"3 {i * n + j} {i * n + (j + 1)} {(i + 1) * n + j}\n")
                f.write(f"3 {(i + 1) * n + j} {i * n + (j + 1)} {(i + 1) * n + (j + 1)}\n")

def write_pos_ori(path: Path, position: ti.template(), orientation: ti.template()):
    # Source orientation uses layout (w, x, y, z)
    q_state = (orientation.x, orientation.y, orientation.z, orientation.w)

    # Move by center of mass rotated into world space
    rotated_cm = rotate_vector(q_state, CENTER_OF_MASS)
    translated_pos = (
        position.x + rotated_cm[0],
        position.y + rotated_cm[1],
        position.z + rotated_cm[2],
    )

    # Apply global +90° rotation about X to both pose and translation
    final_pos = rotate_vector(Q_EXTRA, translated_pos)
    final_q = quat_mul(Q_EXTRA, q_state)

    # Euler angles (XYZ order)
    sinr_cosp = 2.0 * (final_q[0] * final_q[1] + final_q[2] * final_q[3])
    cosr_cosp = 1.0 - 2.0 * (final_q[1] * final_q[1] + final_q[2] * final_q[2])
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (final_q[0] * final_q[2] - final_q[3] * final_q[1])
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (final_q[0] * final_q[3] + final_q[1] * final_q[2])
    cosy_cosp = 1.0 - 2.0 * (final_q[2] * final_q[2] + final_q[3] * final_q[3])
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    final_pos_trans = (
        final_pos[0] * 256.0 + TRANSLATION_OFFSET[0],
        final_pos[1] * 256.0 + TRANSLATION_OFFSET[1],
        final_pos[2] * 256.0 + TRANSLATION_OFFSET[2],
    )

    with path.open("a", encoding="ascii") as f:
        f.write(f"{final_pos_trans[0]}, {final_pos_trans[1]}, {final_pos_trans[2]}, {roll}, {pitch}, {yaw}\n")

def export(frame_count: int, x: ti.template(), n: int, position: ti.template(), orientation: ti.template()):
    path1 = f"cloth/cloth_{frame_count:05d}.ply"
    path2 = f"cloth/bunny_pos_ori.txt"
    write_ply(Path(path1), x, n)
    write_pos_ori(Path(path2), position, orientation)