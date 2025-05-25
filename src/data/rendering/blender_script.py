"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
import datetime
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def debug_print(*args, **kwargs):
    """Enhanced print function that ensures output is flushed immediately and logged to a file."""
    # Print to stdout with immediate flush
    print(*args, **kwargs, flush=True)
    
    # Also log to a file in the output directory if we have args.output_dir
    try:
        if 'args' in globals() and hasattr(args, 'output_dir'):
            log_file = os.path.join(args.output_dir, "blender_debug.log")
            with open(log_file, "a") as f:
                print(*args, **kwargs, file=f)
    except Exception as e:
        print(f"Error logging to file: {e}", flush=True)


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def _sample_spherical_northern_hemisphere() -> np.ndarray:
    """Samples a point in the northern hemisphere of a sphere."""
    theta = random.random() * 2 * math.pi
    phi = math.acos(random.random())
    return np.array([
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
    ])


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = True,
    specular_factor: float = 1.0,
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
):
    """Creates a physically-based light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy/intensity of the light in physical units.
        use_shadow (bool, optional): Whether to use shadows. Defaults to True.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.
        color (Tuple[float, float, float, float], optional): RGBA color. Defaults to white.

    Returns:
        bpy.types.Object: The light object.
    """
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    
    # Physical light properties
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    light_data.color = color[:3]  # RGB only, no alpha
    
    # Additional physical properties based on light type
    if light_type == 'AREA':
        light_data.shape = 'RECTANGLE'
        light_data.size = 1.0
        light_data.size_y = 1.0
    elif light_type == 'SPOT':
        light_data.spot_size = math.radians(45.0)  # 45 degree spot angle
        light_data.spot_blend = 0.15  # Soft edge
    
    return light_object


def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Creates a physically-based lighting setup with exposure control.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene.
    """
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Set exposure for the scene - this helps normalize overall brightness
    scene = bpy.context.scene
    if scene.render.engine == 'CYCLES':
        scene.view_settings.exposure = 0.0  # Baseline exposure
        scene.view_settings.gamma = 1.0
    else:  # EEVEE
        # For EEVEE, we'll control exposure through light intensity
        pass

    # Create physically based key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),  # 45 degrees rotation
        energy=3.0,  # Fixed baseline energy
        use_shadow=True,
    )
    
    # Create physically based fill light (softer, less intense)
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),  # About 135 degrees
        energy=1.5,  # Half the key light energy
        use_shadow=True,
    )
    
    # Create physically based rim light (for edge highlighting)
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),  # About 225 degrees
        energy=2.5,  # Slightly less than key light
        use_shadow=False,  # No shadows for rim light to keep it clean
    )
    
    # Create physically based bottom light (for fill shadows)
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),  # 180 degrees (from bottom)
        energy=1.0,  # Subtle fill light
        use_shadow=False,
    )
    
    # If using Cycles, set up additional physical properties
    if scene.render.engine == 'CYCLES':
        for light in [key_light, fill_light, rim_light, bottom_light]:
            light.data.use_nodes = True
            nodes = light.data.node_tree.nodes
            # Access the emission node that controls the light
            emission = nodes.get('Emission')
            if emission:
                # For sun lights, we don't need to adjust strength further
                # The energy parameter already sets the appropriate value
                pass
                
    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        #  Don't support usdz for now
        return None
        
    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        # For GLTF/GLB, make sure to import with textures
        import_function(filepath=object_path, merge_vertices=True)
    elif file_extension == "obj":
        # For OBJ, ensure MTL files are loaded
        import_function(filepath=object_path)
    else:
        import_function(filepath=object_path)
        
    # --- Apply corrective rotation for GSO standard orientation ---
    # GSO objects are observed to be oriented with their "top" along Blender's -Y axis.
    # Rotate by -90 degrees around the X-axis to make them Z-up.
    
    bpy.ops.object.select_all(action='DESELECT')
    objects_to_rotate = []
    for obj in bpy.data.objects: # Iterate over all objects
        # Check if the object is a MESH and not a camera or light to avoid rotating those
        if obj.type == 'MESH':
            obj.select_set(True) # Select mesh objects
            objects_to_rotate.append(obj)
        elif obj.type not in {'CAMERA', 'LIGHT'}: # Also select other relevant object types if any
            obj.select_set(True)
            objects_to_rotate.append(obj)
        else:
            obj.select_set(False) # Ensure cameras/lights are not selected

    if objects_to_rotate:
        # Ensure there's an active object if we have selection, needed for operators
        if bpy.context.view_layer.objects.active is None and objects_to_rotate:
             bpy.context.view_layer.objects.active = objects_to_rotate[0]
        
        debug_print(f"Applying -90deg X-axis corrective rotation to {len(objects_to_rotate)} selected objects.")
        # Rotate selected objects around the GLOBAL X-axis
        bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X', orient_type='GLOBAL', constraint_axis=(True, False, False))
        
        # Apply the rotation to make it the new base orientation
        # Make sure all objects to rotate are selected before applying transform
        bpy.ops.object.select_all(action='DESELECT')
        for obj_to_apply in objects_to_rotate:
            obj_to_apply.select_set(True)
        if objects_to_rotate: # Ensure active object is set if any selected
            bpy.context.view_layer.objects.active = objects_to_rotate[0]
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            debug_print(f"Corrective rotation applied and baked for selected objects.")
    else:
        debug_print("No suitable objects found to apply corrective rotation.")
    # --- End corrective rotation ---
        
    # After importing, print info about materials for debugging
    print(f"Loaded object with {len(bpy.data.materials)} materials")
    for i, mat in enumerate(bpy.data.materials):
        print(f"Material {i}: {mat.name}, Use Nodes: {mat.use_nodes}")
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    print(f"  - Image texture: {node.image.name}, filepath: {node.image.filepath}")


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.
    """
    # Ensure we're in object mode
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Get all objects that should be normalized (not cameras or lights)
    objects_to_normalize = [obj for obj in bpy.context.scene.objects 
                           if obj.type not in {'CAMERA', 'LIGHT', 'EMPTY'}]
    
    if len(objects_to_normalize) == 0:
        debug_print("WARNING: No objects found in scene to normalize!")
        return
    
    # Step 1: Calculate the bounds of the entire scene
    bbox_min, bbox_max = scene_bbox()
    scene_center = (bbox_min + bbox_max) / 2
    size = bbox_max - bbox_min
    max_dim = max(size.x, size.y, size.z)
    
    if max_dim < 0.0001:
        debug_print("WARNING: Scene has effectively zero size. Using default scale.")
        scale_factor = 1.0
    else:
        scale_factor = 1.0 / max_dim
    
    debug_print(f"Initial bounds: min={bbox_min}, max={bbox_max}, center={scene_center}")
    debug_print(f"Scale factor: {scale_factor}")
    
    # Create a temporary parent for all objects to ensure uniform scaling and translation
    temp_parent = bpy.data.objects.new("TempScaleParent", None)
    bpy.context.collection.objects.link(temp_parent)
    
    # Store original parent relationships and link all objects to the temp parent
    original_parents = {}
    for obj in objects_to_normalize:
        original_parents[obj] = obj.parent
        # Apply any pending transforms to avoid issues
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # Parent to the temp object WITHOUT changing the visual transform
        obj.parent = temp_parent
    
    # Scale the parent (this affects all children uniformly without changing their relative positions)
    temp_parent.scale = Vector((scale_factor, scale_factor, scale_factor))
    # Move to center
    temp_parent.location = -scene_center * scale_factor
    
    # Update the scene
    bpy.context.view_layer.update()
    
    # Apply transformations while preserving relative positions
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects_to_normalize:
        obj.select_set(True)
    temp_parent.select_set(True)
    bpy.context.view_layer.objects.active = temp_parent
    
    # This preserves visual transforms when clearing parent
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    
    # Clean up - remove the temporary parent
    bpy.ops.object.select_all(action='DESELECT')
    temp_parent.select_set(True)
    bpy.ops.object.delete()
    
    # Verify the result
    bpy.context.view_layer.update()
    v_min, v_max = scene_bbox()
    v_center = (v_min + v_max) / 2
    v_size = v_max - v_min
    
    debug_print(f"After normalization: min={v_min}, max={v_max}, center={v_center}")
    debug_print(f"Normalized size: {v_size}")
    
    # Emergency correction - do direct transformation if still not centered
    if v_center.length > 0.01 or max(v_size) > 1.1:
        debug_print(f"Emergency correction needed. Center at {v_center}, size={v_size}")
        
        # Apply direct correction to each object
        correction_scale = 1.0
        if max(v_size) > 1.1:
            correction_scale = 1.0 / max(v_size)
            debug_print(f"Applying emergency scaling: {correction_scale}")
        
        for obj in objects_to_normalize:
            if correction_scale != 1.0:
                obj.scale = obj.scale * correction_scale
            obj.location = obj.location - v_center
        
        # Final verification
        bpy.context.view_layer.update()
        final_min, final_max = scene_bbox()
        final_center = (final_min + final_max) / 2
        final_size = final_max - final_min
        debug_print(f"FINAL CHECK: Center at {final_center}, size={final_size}")
    
    # Make sure the camera is not parented to any object
    camera = bpy.data.objects.get("Camera")
    if camera and camera.parent:
        camera.parent = None


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }


def ensure_texture_visibility() -> None:
    """Ensures that textures are visible in the render by properly configuring material settings."""
    texture_count = 0
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
            
        # Check if there are any texture nodes
        has_textures = False
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                has_textures = True
                texture_count += 1
                
                # Ensure the texture is connected to the output
                principled = None
                for n in material.node_tree.nodes:
                    if n.type == 'BSDF_PRINCIPLED':
                        principled = n
                        break
                
                if principled:
                    # Try to connect the texture to the base color if not already connected
                    if not any(link.to_socket == principled.inputs['Base Color'] for link in material.node_tree.links):
                        material.node_tree.links.new(
                            node.outputs['Color'], 
                            principled.inputs['Base Color']
                        )
    
    print(f"Found and ensured visibility for {texture_count} textures")
    return texture_count > 0


def get_camera_positions(
    num_renders: int,
    only_northern_hemisphere: bool,
    elevation_options: Optional[List[float]] = None,
    azimuth_options: Optional[List[float]] = None,
) -> List[Tuple[float, float, float]]:
    """Returns a list of camera positions for rendering."""
    # If specific angles are provided, use them
    if azimuth_options is not None and elevation_options is not None:
        # Make sure we have enough angles for the requested renders
        if len(azimuth_options) < num_renders or len(elevation_options) < num_renders:
            print(f"Warning: Not enough angles provided. Repeating as needed.")
            # Repeat the angles if we need more
            azimuth_options = list(azimuth_options) * (num_renders // len(azimuth_options) + 1)
            elevation_options = list(elevation_options) * (num_renders // len(elevation_options) + 1)
        
        positions = []
        radius = 1.8  # Fixed camera distance
        
        # Use the provided angles to calculate camera positions
        for i in range(num_renders):
            # Negate the azimuth angle to reverse rotation direction (counter-clockwise)
            azimuth = -math.radians(azimuth_options[i])
            elevation = math.radians(elevation_options[i])
            
            # Convert spherical coordinates to Cartesian
            x = radius * math.cos(elevation) * math.cos(azimuth)
            y = radius * math.cos(elevation) * math.sin(azimuth)
            z = radius * math.sin(elevation)
            
            positions.append((x, y, z))
        
        return positions
    
    # Fall back to the existing random sampling approach
    positions = []
    for _ in range(num_renders):
        if only_northern_hemisphere:
            pos = _sample_spherical_northern_hemisphere() * 1.8  # Apply consistent radius
        else:
            pos = _sample_spherical() 
        positions.append(tuple(pos))
    
    return positions


def render_object(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the input parameters
    debug_print(f"render_object called with num_renders={num_renders}")
    debug_print(f"Object file: {object_file}")
    debug_print(f"Output directory: {output_dir}")
    debug_print(f"Only northern hemisphere: {only_northern_hemisphere}")

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        load_object(object_file)

    # Set up cameras
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # Setup physical render settings
    setup_physical_render_settings()

    # Extract the metadata
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures

    # Make sure textures are visible in the render
    has_visible_textures = ensure_texture_visibility()
    metadata["has_visible_textures"] = has_visible_textures

    # possibly apply a random color to all objects
    if (object_file.endswith(".stl") or object_file.endswith(".ply") or not has_visible_textures):
        # Only apply random colors if there are no textures or for files that don't typically have textures
        print(f"Applying random colors to {object_file} (has_visible_textures={has_visible_textures})")
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None
        print(f"Using existing textures for {object_file}")

    # save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # normalize the scene so that the object fits in the [-0.5, 0.5]^3 cube
    # Use align_to_world=False to preserve original orientation
    normalize_scene()

    # Log to verify normalization worked correctly
    bbox_min, bbox_max = scene_bbox()
    print(f"After normalization - Bounding box min: {bbox_min}, max: {bbox_max}, center: {(bbox_min + bbox_max)/2}")

    # Save the Blender scene for debugging
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_dir, "scene.blend"))

    # Set transparent background
    set_transparent_background()

    # Create physically-based lighting
    lights = randomize_lighting()
    
    # pre-defined angles for 360-degree views
    if num_renders == 12:
        azimuths = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]).astype(float)
        elevations = np.array([20, -10, 20, -10, 20, -10, 20, -10, 20, -10, 20, -10]).astype(float)
    elif num_renders == 8:
        azimuths = np.array([0, 45, 90, 135, 180, 225, 270, 315]).astype(float)
        elevations = np.array([20, -10, 20, -10, 20, -10, 20, -10]).astype(float)
    elif num_renders == 6:
        azimuths = np.array([30, 90, 150, 210, 270, 330]).astype(float)
        elevations = np.array([20, -10, 20, -10, 20, -10]).astype(float)
    
    # Get the camera positions based on specified angles
    camera_positions = get_camera_positions(
        num_renders=num_renders,
        only_northern_hemisphere=only_northern_hemisphere,
        elevation_options=elevations,
        azimuth_options=azimuths
    )

    for i, pos in enumerate(camera_positions):
        # Set camera position
        cam.location = pos
        
        # Get the camera's world matrix
        # This captures the transformation from camera space to world space
        camera_matrix = np.array(cam.matrix_world)
        
        # Save the camera matrix
        np.save(os.path.join(output_dir, f"{i:03d}.npy"), camera_matrix)
        
        # Set output path
        scene.render.filepath = os.path.join(output_dir, f"{i:03d}.png")
        
        # Render the image
        bpy.ops.render.render(write_still=True)


def set_transparent_background() -> None:
    """Sets the background to be transparent for rendering."""
    # Configure render settings for transparency
    bpy.context.scene.render.film_transparent = True
    
    # Access the world settings
    world = bpy.data.worlds['World']
    
    # Use nodes to set up the background
    world.use_nodes = True
    bg_node = world.node_tree.nodes['Background']
    
    # Set the background color with alpha of 0 for transparency
    bg_node.inputs[0].default_value = (1, 1, 1, 0)  # RGBA format with 0 alpha
    
    # For Cycles, make sure the background is handled properly
    world.cycles_visibility.camera = False


def setup_physical_render_settings():
    """Configure render settings for physically-based rendering."""
    scene = bpy.context.scene
    render = scene.render
    
    # Common settings for both engines
    render.use_motion_blur = False
    
    if render.engine == 'CYCLES':
        # Cycles specific settings
        scene.cycles.use_animated_seed = False
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 4
        scene.cycles.transparent_max_bounces = 8
        
        # Color management for realistic rendering
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'None'
    else:
        # EEVEE specific settings
        scene.eevee.taa_render_samples = 32
        scene.eevee.use_gtao = True
        scene.eevee.use_bloom = False
        scene.eevee.use_ssr = True
        scene.eevee.use_ssr_refraction = True
        
        # Color management
        scene.view_settings.view_transform = 'Standard'


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--object_path",
            type=str,
            required=True,
            help="Path to the object file",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            required=True,
            help="Path to the directory where the rendered images and metadata will be saved.",
        )
        parser.add_argument(
            "--engine",
            type=str,
            default="BLENDER_EEVEE",
            choices=["CYCLES", "BLENDER_EEVEE"],
        )
        parser.add_argument(
            "--only_northern_hemisphere",
            action="store_true",
            help="Only render the northern hemisphere of the object.",
            default=False,
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print verbose output",
        )
        
        # Write a marker file to show script was started
        argv = sys.argv[sys.argv.index("--") + 1 :]
        args = parser.parse_args(argv)
        
        # Create a debug log file
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "blender_debug.log"), "w") as f:
            f.write(f"Blender script started at {datetime.datetime.now()}\n")
            f.write(f"Args: {args}\n")
            f.write(f"Python version: {sys.version}\n")
            f.write(f"Blender version: {bpy.app.version_string}\n")
        
        print(f"Script started with args: {args}")
        print(f"Output directory: {args.output_dir}")
        
        context = bpy.context
        scene = context.scene
        render = scene.render

        # Set render settings
        render.engine = args.engine
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode = "RGBA"
        render.resolution_x = 1024
        render.resolution_y = 1024
        render.resolution_percentage = 100
        
        # Set transparent background
        set_transparent_background()
        
        # Additional settings for better texture rendering
        if render.engine == "CYCLES":
            scene.cycles.use_denoising = True
            scene.cycles.samples = 128  # More samples for better quality
        else:  # BLENDER_EEVEE
            scene.eevee.use_ssr = True  # Screen Space Reflections
            scene.eevee.use_ssr_refraction = True
            scene.eevee.taa_render_samples = 32  # Anti-aliasing samples
        
        # Write another marker
        with open(os.path.join(args.output_dir, "before_render.txt"), "w") as f:
            f.write("About to start rendering\n")
        
        # Random num_renders 6, 8 or 12
        num_renders = random.choice([6, 8, 12])
        debug_print(f"Selected random number of renders: {num_renders}")
        
        # Save the chosen number to a file for main.py to read
        with open(os.path.join(args.output_dir, "num_renders.txt"), "w") as f:
            f.write(str(num_renders))
        
        # Render the images
        render_object(
            object_file=args.object_path,
            num_renders=num_renders,
            only_northern_hemisphere=args.only_northern_hemisphere,
            output_dir=args.output_dir,
        )
        
        # Load metadata to extract color information for the completion marker
        metadata_path = os.path.join(args.output_dir, "metadata.json")
        color_info = "Unknown"
        texture_info = "Unknown"
        scene_info = "Unknown"
        camera_info = []
        light_info = []
        render_stats = {}
        
        try:
            # Capture render statistics before they're lost
            render_stats = {
                "resolution": f"{render.resolution_x}x{render.resolution_y}",
                "percentage": f"{render.resolution_percentage}%",
                "engine": args.engine,
                "num_renders": num_renders  # Include the random number in stats
            }
            
            if args.engine == "CYCLES":
                render_stats["samples"] = scene.cycles.samples
                render_stats["denoising"] = "Enabled" if scene.cycles.use_denoising else "Disabled"
            else:  # BLENDER_EEVEE
                render_stats["aa_samples"] = scene.eevee.taa_render_samples
                render_stats["ssr"] = "Enabled" if scene.eevee.use_ssr else "Disabled"
                render_stats["ssr_refraction"] = "Enabled" if scene.eevee.use_ssr_refraction else "Disabled"
            
            # Capture light information
            for light in [obj for obj in scene.objects if obj.type == 'LIGHT']:
                light_data = light.data
                light_info.append({
                    "name": light.name,
                    "type": light_data.type,
                    "energy": light_data.energy,
                    "color": [round(c*255) for c in light_data.color],
                    "location": [round(c, 3) for c in light.location],
                    "rotation": [round(math.degrees(r), 1) for r in light.rotation_euler]
                })
            
            # Capture camera information for the last camera position
            if scene.camera:
                cam = scene.camera
                camera_info = {
                    "name": cam.name,
                    "location": [round(c, 3) for c in cam.location],
                    "rotation": [round(math.degrees(r), 1) for r in cam.rotation_euler],
                    "lens": round(cam.data.lens, 1),
                    "sensor_width": cam.data.sensor_width
                }
            
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                # Get information about random colors
                random_color = metadata.get("random_color")
                has_visible_textures = metadata.get("has_visible_textures", False)
                
                if random_color:
                    # Format the color as RGB values
                    color_values = [round(c * 255) for c in random_color[:3]]
                    color_info = f"Applied random color RGB{tuple(color_values)}"
                    
                    # Add reason for random color
                    if args.object_path.endswith(".stl") or args.object_path.endswith(".ply"):
                        color_info += " (STL/PLY format doesn't support textures)"
                    elif not has_visible_textures:
                        color_info += " (No visible textures found in model)"
                else:
                    color_info = "Used original model textures"
                    
                # Add texture information
                texture_info = f"Model has visible textures: {has_visible_textures}"
                
                # Get material count for additional context
                material_count = metadata.get("material_count", 0)
                texture_info += f", Material count: {material_count}"
                
                # Add scene information
                scene_info = {
                    "poly_count": metadata.get("poly_count", "Unknown"),
                    "vert_count": metadata.get("vert_count", "Unknown"),
                    "bbox_dimensions": "Unknown"
                }
                
                # Add bounding box dimensions if available
                if "scene_size" in metadata and "bbox_min" in metadata["scene_size"] and "bbox_max" in metadata["scene_size"]:
                    bbox_min = metadata["scene_size"]["bbox_min"]
                    bbox_max = metadata["scene_size"]["bbox_max"]
                    dimensions = [round(bbox_max[i] - bbox_min[i], 3) for i in range(3)]
                    scene_info["bbox_dimensions"] = f"X: {dimensions[0]}, Y: {dimensions[1]}, Z: {dimensions[2]}"
                    scene_info["bbox_center"] = [round((bbox_max[i] + bbox_min[i])/2, 3) for i in range(3)]
        
        except Exception as e:
            print(f"Error gathering detailed info: {str(e)}")
            # Continue even if detailed info gathering fails
        
        # Final marker with enhanced information
        with open(os.path.join(args.output_dir, "render_complete.txt"), "w") as f:
            f.write("Rendering completed successfully\n\n")
            
            f.write("=== OBJECT INFORMATION ===\n")
            f.write(f"Object path: {args.object_path}\n")
            f.write(f"Color rendering: {color_info}\n")
            f.write(f"Texture information: {texture_info}\n\n")
            
            f.write("=== SCENE INFORMATION ===\n")
            if isinstance(scene_info, dict):
                f.write(f"Poly count: {scene_info['poly_count']}\n")
                f.write(f"Vertex count: {scene_info['vert_count']}\n")
                f.write(f"Bounding box dimensions: {scene_info['bbox_dimensions']}\n")
                if "bbox_center" in scene_info:
                    f.write(f"Bounding box center: {scene_info['bbox_center']}\n\n")
            else:
                f.write(f"{scene_info}\n\n")
                
            f.write("=== CAMERA INFORMATION ===\n")
            if camera_info:
                f.write(f"Camera name: {camera_info['name']}\n")
                f.write(f"Position (XYZ): {camera_info['location']}\n")
                f.write(f"Rotation (degrees): {camera_info['rotation']}\n")
                f.write(f"Focal length: {camera_info['lens']}mm\n")
                f.write(f"Sensor width: {camera_info['sensor_width']}mm\n\n")
            else:
                f.write("Camera information not available\n\n")
                
            f.write("=== LIGHTING INFORMATION ===\n")
            if light_info:
                for i, light in enumerate(light_info):
                    f.write(f"Light {i+1}: {light['name']} ({light['type']})\n")
                    f.write(f"  Energy: {light['energy']}\n")
                    f.write(f"  Color (RGB): {light['color']}\n")
                    f.write(f"  Position: {light['location']}\n")
                    f.write(f"  Rotation (degrees): {light['rotation']}\n")
                f.write("\n")
            else:
                f.write("Lighting information not available\n\n")
                
            f.write("=== RENDER SETTINGS ===\n")
            f.write(f"Engine: {render_stats.get('engine', args.engine)}\n")
            f.write(f"Resolution: {render_stats.get('resolution', 'Unknown')}\n")
            if 'samples' in render_stats:
                f.write(f"Samples: {render_stats['samples']}\n")
                f.write(f"Denoising: {render_stats['denoising']}\n")
            if 'aa_samples' in render_stats:
                f.write(f"AA Samples: {render_stats['aa_samples']}\n")
                f.write(f"Screen Space Reflections: {render_stats['ssr']}\n")
                f.write(f"SSR Refraction: {render_stats['ssr_refraction']}\n")
            f.write("\n")
            
            f.write(f"Rendering completed at: {datetime.datetime.now().isoformat()}\n")
            
    except Exception as e:
        # Write error to a file in output directory
        error_path = os.path.join(args.output_dir if 'args' in locals() else "/tmp", "blender_error.txt")
        with open(error_path, "w") as f:
            f.write(f"Error in blender script: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        print(f"ERROR: {str(e)}")
        sys.exit(1)