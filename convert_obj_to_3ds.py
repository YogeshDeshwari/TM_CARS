"""
Blender script to convert OBJ to 3DS

Usage:
  blender --background --python convert_obj_to_3ds.py -- input.obj output.3ds
"""
import bpy
import sys
import os

# Enable 3DS addon if available
try:
    bpy.ops.preferences.addon_enable(module='io_scene_3ds')
    print("Enabled io_scene_3ds addon")
except:
    pass

# Get arguments
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    print("Usage: blender --background --python convert_obj_to_3ds.py -- input.obj output.3ds")
    sys.exit(1)

input_path = os.path.abspath(argv[0])
output_path = os.path.abspath(argv[1])

print(f"\n{'='*60}")
print("OBJ to 3DS Converter")
print(f"{'='*60}")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ
print("\nImporting OBJ...")
try:
    bpy.ops.wm.obj_import(filepath=input_path)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=input_path)

# Count and select mesh objects
mesh_count = 0
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.select_set(True)
        mesh_count += 1

print(f"Imported {mesh_count} mesh objects")

# Export to 3DS
print("\nExporting to 3DS...")
exported = False

# Try different export methods
try:
    bpy.ops.export_scene.autodesk_3ds(filepath=output_path, use_selection=True)
    exported = True
    print(f"SUCCESS: Exported to {output_path}")
except Exception as e:
    print(f"Method 1 failed: {e}")

if not exported:
    try:
        # Try io_scene_3ds addon using getattr to avoid syntax error
        export_op = getattr(bpy.ops.wm, '3ds_export', None)
        if export_op:
            export_op(filepath=output_path)
            exported = True
            print(f"SUCCESS: Exported to {output_path}")
        else:
            print("Method 2: 3ds_export not available")
    except Exception as e:
        print(f"Method 2 failed: {e}")

if not exported:
    # Export as FBX as fallback (TMNF might accept it)
    fbx_path = output_path.replace('.3ds', '.fbx')
    try:
        bpy.ops.export_scene.fbx(filepath=fbx_path, use_selection=True)
        print(f"Exported as FBX: {fbx_path}")
        exported = True
    except Exception as e:
        print(f"FBX export failed: {e}")

if not exported:
    print("\nERROR: Could not export to any format!")
    print("Please install the 3DS export addon in Blender")
else:
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")

