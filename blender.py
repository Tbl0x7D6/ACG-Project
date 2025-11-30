import bpy
import bmesh
from math import radians

def remesh():
    bpy.ops.object.mode_set(mode='OBJECT')

    # 1. Select object
    target_object = None
    for obj in bpy.data.objects:
        if obj.name == "mesh_sequence" and obj.type == 'MESH':
            target_object = obj
            break
    bpy.ops.object.select_all(action='DESELECT')
    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object
    
    mesh = target_object.data
    
    # 2. First remesh
    mesh.remesh_mode = 'VOXEL'
    mesh.remesh_voxel_size = 0.3
    mesh.remesh_voxel_adaptivity = 0.3
    mesh.use_remesh_preserve_volume = True
    mesh.use_remesh_preserve_attributes = True
    bpy.ops.object.voxel_remesh()
    
    # 3. Second remesh
    mesh.remesh_voxel_adaptivity = 1.0
    bpy.ops.object.voxel_remesh()
    
    # 4. Shade auto smooth
    bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=radians(90))
    
    # 5. Assign material
    material = bpy.data.materials.get("Material.010")
    target_object.data.materials.clear()
    target_object.data.materials.append(material)
    
    return True

def main():
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        success = remesh()
        if not success:
            print(f"Remeshing failed at frame {frame}")
            break

main()