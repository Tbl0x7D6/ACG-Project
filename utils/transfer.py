import json
import math
import sys
import json

positions = []
angular_positions = []

with open("rigid_states/config.json", "r") as f:
    scale_data = json.load(f)
scale = scale_data["scale"]

# Center of mass in simulation space (same frame as rigid_body.py)
CENTER_OF_MASS = (0.037939, 0.065785, -0.038762)
CENTER_OF_MASS_SCALED = tuple(c * 256.0 * scale / 0.3 for c in CENTER_OF_MASS)

# Quaternion multiply with layout (w, x, y, z) as used in rigid_body.py
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


# Extra global rotations applied after the simulation pose
HALF_ANGLE = 0.25 * math.pi  # 90° / 2
Q_X_90 = (math.cos(HALF_ANGLE), math.sin(HALF_ANGLE), 0.0, 0.0)
Q_Z_90 = (math.cos(HALF_ANGLE), 0.0, 0.0, math.sin(HALF_ANGLE))
Q_EXTRA = quat_mul(Q_Z_90, Q_X_90)  # Rz(90°) * Rx(90°) left-multiplied


def main(index_start, index_end):
    for i in range(index_start, index_end):
        with open(f"rigid_states/state_{i:05d}.json", "r") as f:
            data = json.load(f)

            # State quaternions are stored as (w, x, y, z)
            qw, qx, qy, qz = data["quaternion"]

            q_state = (qw, qx, qy, qz)

            pos_world = (
                data["position"][0] * 256.0,
                data["position"][1] * 256.0,
                data["position"][2] * 256.0,
            )

            rotated_cm = rotate_vector(q_state, CENTER_OF_MASS_SCALED)

            # Equivalent single translation after T(cm) -> R(q) -> T(pos)
            final_translation = (
                pos_world[0] + rotated_cm[0],
                pos_world[1] + rotated_cm[1],
                pos_world[2] + rotated_cm[2],
            )

            # Apply the extra global rotations to both orientation and translation
            final_pos = rotate_vector(Q_EXTRA, final_translation)
            # !!!!!!!!!!!!!!!!  REMEMBER TO MINUS 256 FROM Y COORDINATE  !!!!!!!!!!!!!!!!
            final_pos = (final_pos[0], final_pos[1] - 256, final_pos[2]) 
            final_q = quat_mul(Q_EXTRA, q_state)

            # roll (X), pitch (Y), yaw (Z) from quaternion (XYZ order) after extra rotations
            sinr_cosp = 2.0 * (final_q[0] * final_q[1] + final_q[2] * final_q[3])
            cosr_cosp = 1.0 - 2.0 * (final_q[1] * final_q[1] + final_q[2] * final_q[2])
            roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (final_q[0] * final_q[2] - final_q[3] * final_q[1])
            pitch = math.asin(max(-1.0, min(1.0, sinp)))

            siny_cosp = 2.0 * (final_q[0] * final_q[3] + final_q[1] * final_q[2])
            cosy_cosp = 1.0 - 2.0 * (final_q[2] * final_q[2] + final_q[3] * final_q[3])
            yaw = math.atan2(siny_cosp, cosy_cosp)

            positions.append(final_pos)
            angular_positions.append((roll, pitch, yaw))

    with open("rigid_states/final.txt", "w") as f:
        f.write(f"{scale}\n")
        for index in range(len(positions)):
            pos = positions[index]
            ang = angular_positions[index]
            f.write(f"{pos[0]}, {pos[1]}, {pos[2]}, {ang[0]}, {ang[1]}, {ang[2]}\n")


if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    main(start, end)