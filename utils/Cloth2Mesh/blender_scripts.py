import bpy
import bmesh
from math import radians

def remesh():
    bpy.ops.object.mode_set(mode='OBJECT')

    # 1. Select object
    target_object = None
    for obj in bpy.data.objects:
        if obj.name == "cloth_sequence" and obj.type == 'MESH':
            target_object = obj
            break
    bpy.ops.object.select_all(action='DESELECT')
    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object
    
    # 2. Shade auto smooth
    bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=radians(30))
    
    # 3. Assign material
    material = bpy.data.materials.get("Fabric")
    target_object.data.materials.clear()
    target_object.data.materials.append(material)
    
    return True

def set_cloth_material():
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
    
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(frames_data)

if __name__ == "__main__":
    set_cloth_material()
    file_path = "E:/taichi/ACG-Project/output/bunny_pos_ori.txt"
    set_bunny_animation_from_file(file_path)