def loader(mesh_file, scale=1.0):
    print("[DEBUG] Loading mesh from: ", mesh_file, " with scale: ", scale, end='')
    lines = []
    with open(mesh_file, 'r') as f:
        lines = f.readlines()
    _faces = []
    _vertices = []
    for line in lines:
        if line[0] == 'v':
            parts = line.split()
            vertex = [float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale]
            _vertices.append(vertex)
        elif line[0] == 'f':
            parts = line.split()
            face = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]
            _faces.append(face)
    print(" - Done. Loaded", len(_vertices), "vertices and", len(_faces), "faces.")
    return _vertices, _faces