import json
import math

positions = []
angular_positions = []


# Center of mass in simulation space (Taichi uses w, x, y, z quaternions)
CENTER_OF_MASS = (-0.037939, -0.065785, 0.038762)
CENTER_OF_MASS_SCALED = tuple(c * -256.0 for c in CENTER_OF_MASS)


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
    # Rotate a 3D vector v by quaternion q
    q_v = (0.0, v[0], v[1], v[2])
    return quat_mul(quat_mul(q, q_v), quat_conj(q))[1:]


for i in range(600):
    with open(f"../rigid_states/state_{i:05d}.json", "r") as f:
        data = json.load(f)

        qw, qx, qy, qz = data["quaternion"]

        # roll (X), pitch (Y), yaw (Z) from quaternion (XYZ order)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        angular_positions.append((roll, pitch, yaw))

        pos_world = (
            data["position"][0] * 256.0,
            data["position"][1] * 256.0,
            data["position"][2] * 256.0,
        )

        rotated_cm = rotate_vector((qw, qx, qy, qz), CENTER_OF_MASS_SCALED)

        # Equivalent single translation after T(cm) -> R(q) -> T(pos)
        final_translation = (
            pos_world[0] + rotated_cm[0],
            pos_world[1] + rotated_cm[1],
            pos_world[2] + rotated_cm[2],
        )

        positions.append(final_translation)

with open("../rigid_states/final.txt", "w") as f:
    for index in range(len(positions)):
        pos = positions[index]
        ang = angular_positions[index]
        f.write(f"{pos[0]}, {pos[1]}, {pos[2]}, {ang[0]}, {ang[1]}, {ang[2]}\n")