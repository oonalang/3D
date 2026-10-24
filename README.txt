CHARACTER RECONSTRUCTION, 360-DEGREE APPROXIMATION

FILE GUIDE
- Character_Rebuilt_360_Approx.glb: preferred single-file mesh with image texture and colored 3D surfaces. Uses KHR_materials_unlit on the front image-matched part.
- OBJ_editable: traditional OBJ plus MTL and source image texture; vertex colors on non-photo surfaces depend on OBJ viewer support.
- front.png, three_quarter.png, side.png, rear.png: renders showing what this model looks like from each angle.
- preview_grid.jpg: four views together.
- rebuild_character.py: procedural construction script for making further changes.

WHAT WAS BUILT
- Reference-based face/front texture projection, maintained separately from rear shapes.
- New approximate rear head/body/limb surfaces, plus layered hair geometry, a curved tail and horns.
- Rear-facing surface and garment details instead of a mirrored frontal texture on the back.
- Approximate extra surface colors reduce repeated photo artifacts around the hair.

KNOWN LIMITATIONS
- A single front photo cannot specify exact side or back appearance; those are interpretations.
- The supplied source OBJ is fragmented (1,170 disconnected mesh islands). Some triangulation, separation, visible artifacts and missing detail remain.
- NOT an exact sculpt of the source illustration, NOT rigged or animation-ready, and not retopologized for production.
- Some apps show vertex colors on the OBJ differently; GLB is preferred.
