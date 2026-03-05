"""
Blender Script: Remove Mudguards from Stadium Car

HOW TO USE:
===========
1. Open Blender
2. Delete default cube (select it, press X, confirm)
3. File -> Import -> Wavefront (.obj) -> Select models/StadiumCar.obj
4. Switch to Scripting workspace (top tabs)
5. Click "New" to create a new script
6. Paste this entire script
7. Click "Run Script" (play button)
8. The mudguards will be selected - press X and choose "Faces" to delete
9. File -> Export -> 3D Studio (.3ds) -> Save as StadiumCar_NoMudguards.3ds

Then use TMNF's importer: Help -> Custom data -> Car geometry
"""

import bpy
import bmesh
from mathutils import Vector

def remove_mudguards():
    # Get the active object (should be the imported car)
    obj = bpy.context.active_object
    
    if obj is None or obj.type != 'MESH':
        # Try to find a mesh object
        for o in bpy.data.objects:
            if o.type == 'MESH':
                obj = o
                break
    
    if obj is None or obj.type != 'MESH':
        print("ERROR: No mesh object found! Import the OBJ first.")
        return
    
    print(f"Working on object: {obj.name}")
    
    # Enter edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Get the bmesh
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    
    # Deselect all first
    bpy.ops.mesh.select_all(action='DESELECT')
    
    # Define mudguard regions based on geometry analysis:
    # Front wheels: Z around 1.4-2.1, |X| > 0.5, Y < 0.55
    # Rear wheels: Z around -1.6 to -0.8, |X| > 0.5, Y < 0.55
    # Also include wheel caps which are at the outer edges
    
    mudguard_faces = []
    wheel_cap_faces = []
    
    for face in bm.faces:
        # Get face center
        center = face.calc_center_median()
        
        # Get all vertex positions for this face
        verts = [v.co for v in face.verts]
        min_x = min(v.x for v in verts)
        max_x = max(v.x for v in verts)
        min_y = min(v.y for v in verts)
        max_y = max(v.y for v in verts)
        min_z = min(v.z for v in verts)
        max_z = max(v.z for v in verts)
        
        # Check if face is in front mudguard region
        # Front mudguards: above front wheels, wrapping around them
        is_front_mudguard = (
            max_z > 1.35 and  # Front of car
            (max_x > 0.55 or min_x < -0.55) and  # Side of car
            max_y < 0.58 and  # Lower part
            min_y > 0.1  # Not the very bottom
        )
        
        # Check if face is in rear mudguard region
        # Rear mudguards: above rear wheels
        is_rear_mudguard = (
            min_z < -0.75 and  # Rear of car
            (max_x > 0.55 or min_x < -0.55) and  # Side of car
            max_y < 0.58 and  # Lower part
            min_y > 0.1  # Not the very bottom
        )
        
        # Wheel caps: the circular caps on wheels
        # Wheels are at |X| around 0.86-0.89, very thin in X direction
        is_wheel_cap = (
            (abs(center.x) > 0.95) and  # At outer edge
            max_y < 0.72 and min_y > 0.0  # Wheel height range
        )
        
        if is_front_mudguard or is_rear_mudguard:
            mudguard_faces.append(face)
        
        if is_wheel_cap:
            wheel_cap_faces.append(face)
    
    print(f"Found {len(mudguard_faces)} mudguard faces")
    print(f"Found {len(wheel_cap_faces)} wheel cap faces")
    
    # Select the mudguard faces
    for face in mudguard_faces:
        face.select = True
    
    for face in wheel_cap_faces:
        face.select = True
    
    # Update the mesh
    bmesh.update_edit_mesh(obj.data)
    
    # Count selected
    selected_count = sum(1 for f in bm.faces if f.select)
    print(f"Total selected faces: {selected_count}")
    print("\n" + "="*50)
    print("FACES SELECTED!")
    print("="*50)
    print("\nNow you can:")
    print("1. Press X and choose 'Faces' to delete them")
    print("2. Or press H to hide them temporarily")
    print("3. After deletion: File -> Export -> 3D Studio (.3ds)")
    print("\nThen in TMNF: Help -> Custom data -> Car geometry")
    print("="*50)


def show_regions_debug():
    """
    Alternative: Just highlight different regions with vertex colors
    for debugging/visualization
    """
    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        for o in bpy.data.objects:
            if o.type == 'MESH':
                obj = o
                break
    
    if obj is None:
        print("No mesh found")
        return
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bm = bmesh.from_edit_mesh(obj.data)
    
    print("\nObject bounds:")
    all_verts = [v.co for v in bm.verts]
    print(f"  X: {min(v.x for v in all_verts):.3f} to {max(v.x for v in all_verts):.3f}")
    print(f"  Y: {min(v.y for v in all_verts):.3f} to {max(v.y for v in all_verts):.3f}")
    print(f"  Z: {min(v.z for v in all_verts):.3f} to {max(v.z for v in all_verts):.3f}")
    
    bpy.ops.object.mode_set(mode='OBJECT')


# Run the main function
print("\n" + "="*50)
print("MUDGUARD REMOVAL SCRIPT")
print("="*50)
remove_mudguards()



