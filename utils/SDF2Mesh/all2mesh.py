from tomesh import export

base_dir = "./tomesh/"
output_dir = "./plys/"
number_of_files = 600

if __name__ == "__main__":
    for i in range(1, number_of_files + 1):
        input_path = f"{base_dir}phi_{i:05d}.npy"
        output_path = f"{output_dir}output_{i:05d}.ply"
        export(input_path, output_path)