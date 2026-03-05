"""
Blender Headless Script: Automatically Remove Mudguards

RUN FROM COMMAND LINE:
======================
blender --background --python blender_auto_remove_mudguards.py -- models/StadiumCar.obj models/StadiumCar_NoMudguards.3ds

OR if you don't have blender in PATH:
/Applications/Blender.app/Contents/MacOS/Blender --background --python blender_auto_remove_mudguards.py -- models/StadiumCar.obj models/StadiumCar_NoMudguards.3ds
"""

import bpy
import bmesh
import sys
import os

def clear_scene():
    """Remove all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_obj(filepath):
    """Import an OBJ file"""
    bpy.ops.wm.obj_import(filepath=filepath)
    return bpy.context.selected_objects[0] if bpy.context.selected_objects else None

def remove_mudguards(obj):
    """Remove mudguard and wheel cap faces from the mesh"""
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    
    faces_to_delete = []
    
    for face in bm.faces:
        verts = [v.co for v in face.verts]
        min_x = min(v.x for v in verts)
        max_x = max(v.x for v in verts)
        min_y = min(v.y for v in verts)
        max_y = max(v.y for v in verts)
        min_z = min(v.z for v in verts)
        max_z = max(v.z for v in verts)
        center = face.calc_center_median()
        
        # Front mudguards
        is_front_mudguard = (
            max_z > 1.35 and
            (max_x > 0.55 or min_x < -0.55) and
            max_y < 0.58 and
            min_y > 0.1
        )
        
        # Rear mudguards
        is_rear_mudguard = (
            min_z < -0.75 and
            (max_x > 0.55 or min_x < -0.55) and
            max_y < 0.58 and
            min_y > 0.1
        )
        
        # Wheel caps (outer edge of wheels)
        is_wheel_cap = (
            abs(center.x) > 0.95 and
            max_y < 0.72 and min_y > 0.0
        )
        
        if is_front_mudguard or is_rear_mudguard or is_wheel_cap:
            faces_to_delete.append(face)
    
    print(f"Deleting {len(faces_to_delete)} faces (mudguards + wheel caps)")
    
    # Delete the faces
    bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
    
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return len(faces_to_delete)

def export_3ds(filepath):
    """Export scene to 3DS format"""
    # Select all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    
    # Export - try different methods based on Blender version
    try:
        bpy.ops.export_scene.autodesk_3ds(filepath=filepath, use_selection=True)
    except AttributeError:
        try:
            bpy.ops.wm.3ds_export(filepath=filepath)
        except:
            print("WARNING: 3DS export not available. Exporting as OBJ instead.")
            obj_path = filepath.replace('.3ds', '.obj')
            bpy.ops.wm.obj_export(filepath=obj_path)
            return obj_path
    
    return filepath

def main():
    # Get command line arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Usage: blender --background --python script.py -- input.obj output.3ds")
        return
    
    if len(argv) < 2:
        print("Usage: blender --background --python script.py -- input.obj output.3ds")
        return
    
    input_path = argv[0]
    output_path = argv[1]
    
    # Make paths absolute
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)
    
    print(f"\n{'='*60}")
    print("MUDGUARD REMOVAL - AUTOMATED")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Clear scene
    clear_scene()
    
    # Import OBJ
    print("\nImporting OBJ...")
    obj = import_obj(input_path)
    if obj is None:
        print("ERROR: Failed to import OBJ")
        return
    
    print(f"Imported: {obj.name}")
    print(f"Vertices: {len(obj.data.vertices)}")
    print(f"Faces: {len(obj.data.polygons)}")
    
    # Remove mudguards
    print("\nRemoving mudguards...")
    deleted = remove_mudguards(obj)
    
    print(f"\nAfter removal:")
    print(f"Vertices: {len(obj.data.vertices)}")
    print(f"Faces: {len(obj.data.polygons)}")
    
    # Export
    print(f"\nExporting to: {output_path}")
    actual_output = export_3ds(output_path)
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nOutput saved to: {actual_output}")
    print("\nNEXT STEPS:")
    print("1. Open TrackMania Forever/United")
    print("2. Go to: Help -> Custom data -> Car geometry")
    print("3. Select the exported file")
    print("4. This will create MainBody.Solid.Gbx")
    print("5. Package with CH_2026 textures")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()



