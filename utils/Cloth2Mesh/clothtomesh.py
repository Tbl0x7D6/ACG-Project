import taichi as ti
from pathlib import Path
import math

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
                f.write(f"{x[i, j].x} {x[i, j].y} {x[i, j].z}\n")

        for i in range(n - 1):
            for j in range(n - 1):
                f.write(f"3 {i * n + j} {i * n + (j + 1)} {(i + 1) * n + j}\n")
                f.write(f"3 {(i + 1) * n + j} {i * n + (j + 1)} {(i + 1) * n + (j + 1)}\n")

def write_pos_ori(path: Path, position: ti.template(), orientation: ti.template()):
    w, x, y, z = orientation.x, orientation.y, orientation.z, orientation.w
    # roll (X axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = ti.atan2(sinr_cosp, cosr_cosp)
    # pitch (Y axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = ti.asin(ti.max(-1.0, ti.min(1.0, sinp)))
    # yaw (Z axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = ti.atan2(siny_cosp, cosy_cosp)

    x = position.x
    y = position.z
    z = position.y

    with path.open("a", encoding="ascii") as f:
        f.write(f"{x}, {y}, {z}, {roll + math.pi / 2}, {pitch}, {yaw}\n")
        # print(f"Written bunny position and orientation to {path}: {position.x}, {position.y}, {position.z}, {roll}, {pitch}, {yaw}\n")

def export(frame_count: int, x: ti.template(), n: int, position: ti.template(), orientation: ti.template()):
    path1 = f"E:/taichi/ACG-Project/output/cloth_{frame_count:05d}.ply"
    path2 = f"E:/taichi/ACG-Project/output/bunny_pos_ori.txt"
    write_ply(Path(path1), x, n)
    write_pos_ori(Path(path2), position, orientation)