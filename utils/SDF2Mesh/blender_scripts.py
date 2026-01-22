import bpy
import bmesh
from math import radians

def remesh():
    bpy.ops.object.mode_set(mode='OBJECT')

    # 1. Select object
    target_object = None
    for obj in bpy.data.objects:
        if obj.name == "output_sequence" and obj.type == 'MESH':
            target_object = obj
            break
    bpy.ops.object.select_all(action='DESELECT')
    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object

    # 2. Cast shadow caustics
    bpy.context.object.cycles.is_caustics_caster = True
    
    # 3. Shade auto smooth
    bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=radians(30))
    
    # 4. Assign material
    material = bpy.data.materials.get("Water")
    target_object.data.materials.clear()
    target_object.data.materials.append(material)
    
    # 5. Assign scale and position
    target_object.scale = (1, 1, -1)
    target_object.location = (0, -256, 0)
    
    return True

def set_material():
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        success = remesh()
        if not success:
            print(f"Remeshing failed at frame {frame}")
            break

def set_bunny_animation_from_file(filepath):
    bunny = bpy.data.objects.get("bunny")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip()
    scale_value = float(first_line) * 256.0
    lines = lines[1:]

    frames_data = []
    for i, line in enumerate(lines):
        line = line.strip()
        try:
            values = [float(v.strip()) for v in line.split(',')]
            x, y, z, rx, ry, rz = values
            frames_data.append((x, y, z, rx, ry, rz))
            
        except ValueError as e:
            print(f"Error parsing line {i + 1}: {line}. Error: {e}")
            continue

    for frame_num, (x, y, z, rx, ry, rz) in enumerate(frames_data, start=1):
        bpy.context.scene.frame_set(frame_num)

        bunny.location = (x, y, z)
        bunny.keyframe_insert(data_path="location", frame=frame_num)
        
        bunny.rotation_euler = (rx, ry, rz)
        bunny.keyframe_insert(data_path="rotation_euler", frame=frame_num)

        bunny.scale = (scale_value, scale_value, scale_value)
        bunny.keyframe_insert(data_path="scale", frame=frame_num)
    
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(frames_data)

if __name__ == "__main__":
    set_material()
    file_path = "../rigid_states/final.txt"
    set_bunny_animation_from_file(file_path)