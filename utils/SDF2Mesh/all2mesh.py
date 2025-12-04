import sys, os
from tomesh import export

if __name__ == "__main__":
    base_dir = sys.argv[1]
    output_dir = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    for i in range(start, end):
        input_path = os.path.join(base_dir, f"phi_{i:05d}.npy")
        output_path = os.path.join(output_dir, f"output_{i:05d}.ply")
        print(f"Processing {input_path} to {output_path}")
        export(input_path, output_path)