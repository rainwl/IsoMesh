# IsoMesh
IsoMesh is a group of related tools for Unity for converting meshes into signed distance field data, raymarching signed distance fields, and extracting signed distance field data back to meshes via surface nets or dual contouring. All the work is parallelized on the GPU using compute shaders.

My motivation for making this was simple: I want to make a game in which I morph and manipulate meshes, and this seemed like the right technique for the job. Isosurface extraction is most often used for stuff like terrain manipulation. Check out No Man's Sky, for example.

I decided to share my code here because it represents a lot of trial and error and research on my part. And frankly, I just think it's cool.

![isomesh2](https://user-images.githubusercontent.com/18707147/115974497-2070d200-a555-11eb-85cd-cfffed99c771.png)

The project is currently being developed and tested on Unity 2020.3.0f1.

## Signed Distance Fields
A signed distance field, or 'SDF', is a function which takes a position in space and returns the distance from that point to the surface of an object. The distance is negative if the point is inside the object. These functions can be used to represent all sorts of groovy shapes, and are in some sense 'volumetric', as opposed to the more traditional polygon-based way of representing geometry.

If you're unfamiliar with SDFs, I would be remiss if I didn't point you to the great [Inigo Quilez](https://www.iquilezles.org/). I'm sure his stuff will do a much better job explaining it than I could.

### Signed Distance Fields + Meshes
While SDFs are really handy, they're mostly good for representing primitive shapes like spheres, cuboids, cones, etc. You can make some pretty impressive stuff just by combining and applying transformations to those forms, but for this project I wanted to try combine the mushy goodness of SDFs with the versatility of triangle meshes. I do this by sampling points in a bounding box around a mesh, and then interpolating the in-between bits.

#### Adding Meshes
In order to add a mesh of your own, open Tools > 'Mesh to SDF'. Give it a mesh reference and select a sample size, I suggest 64. Remember this is cubic, so the workload and resulting file size increases very quickly.

There is also the option to tessellate the mesh before creating the SDF. This will take time and increase the GPU workload, but it will not alter the size of the resulting file. The advantage of the tessellation step is that the resulting polygons will have their positions interpolated according to the normals of the source vertices, turning the 'fake' surfaces of normal interpolation into true geometry. This can produce smoother looking results, but it's usually unnecessary.

If your mesh has UVs you can sample those too. This is currently just experimental: naively sampling UVs has a big assumption built in - namely that your UVs are continuous across ths surface of your mesh. As soon you hit seams you'll see pretty bad artefacts as the UVs rapidly interpolate from one part of your texture to the other.

![isomesh1](https://user-images.githubusercontent.com/18707147/115974173-d686ec80-a552-11eb-8308-87ddec99cd16.png)

## Project Structure
In this project, you'll find two scenes: one for mesh generation and one for raymarching. Both have very similar structures and are just meant to show how to use the tools.

SDF objects are divided into two different components: 'SDFPrimitives' and 'SDFMeshes'. The SDFPrimitive component is standalone and can only currently represent four objects: a sphere, a (rounded) cuboid, a torus, and a box frame. SDFMeshes provide a reference to an SDFMeshAsset file generated by the Mesh to SDF tool. These objects behave much as you'd expect from a Unity GameObject: you can move them around, rotate them, etc. These objects can be set to either 'min' or 'subtract' - min (minimum) objects will combine with others, subtract objects will 'cut holes' in all the objects above them in the hierarchy. For now these are the only SDF primitives and operations I've added, [but there are many more.](https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm) These objects can be added to the scene by right-clicking in the hierarchy.

![isomesh8](https://user-images.githubusercontent.com/18707147/115975686-afceb300-a55e-11eb-9af4-d6315d284f19.png)

You'll always have an 'SDFGroup' as the parent of everything within your system of SDFMeshes and SDFPrimitives. Objects within this system are expected to interact with one another and share the same visual properties such as colour and texture.

The final essential element is an 'SDFGroupRaymarcher' or 'SDFGroupMeshGenerator'. You can have as many of these as you want under one group. SDFGroupMeshGenerators can even represent chunks - though they need to overlap by the size of one cell on all sides, and should have all the same settings.

## Isosurface Extraction
Given a scalar field function, which returns a single value for any point, an isosurface is everywhere the function has the same value. In SDFs, the isosurface is simply the points where the distance to the surface is zero.

Isosurface extraction here refers to converting an isosurface back into a triangle mesh. There are many different algorithms for isosurface extraction, perhaps the most well known being the 'marching cubes' algorithm. In this project, I implemented two (very similar) isosurface extraction algorithms: surface nets and dual contouring. I won't go into any more detail on these algorithms here, [as others have already explained them very well.](https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/)

As I say above, in order to use the isosurface extraction just add an SDFGroupMeshGenerator under an SDFGroup. The number of options on this component is almost excessive, but don't let that get you down, they all have tooltips which do some explaining, and if you've done your homework they should feel fairly familiar:

![isomesh3](https://user-images.githubusercontent.com/18707147/118203187-713c6200-b453-11eb-9438-f105f98f4226.png)

Normal settings are handy to control the appearance of the mesh surface. 'Max angle tolerance' will generate new mesh vertices when normals are too distinct from the normal of their triangle. I like to keep this value around 40 degrees, as it retains sharp edges while keeping smooth curves. 'Visual smoothing' changes the distance between samples when generating mesh normals via central differences.

![isomesh4](https://user-images.githubusercontent.com/18707147/115974786-21a2fe80-a557-11eb-84a0-62c28b537501.png)

There are some additions I've made to try to improve the results of dual contouring. While it produces good edges and corners, QEF can sometimes be a little unstable. I provide optional direct control over the QEF (quadratic error function) constants. I provide two techniques for finding the exact surface intersection points between SDF samples - interpolation is fast but gives kinda poor results at corners. Binary search provides much more exact results but is an iterative solution.

Gradient descent is another iterative improvement which simply moves the vertices back onto the isosurface. Honestly, I see no reason not to always have this on.

The 'nudge' stuff is an experimental solution for reducing artefacts on mesh edges and corners. It moves the mesh vertices a tiny bit in the direction of the sum of the normals at the voxel edge intersections. By itself, this simply 'inflates' the mesh. But alongside gradient descent, it can produce tasty sharp edges, but introduces artefacts of its own. I recommend keeping this value very low if you do use it.

### A note on 'output mode'
You may notice there is an option to switch between 'Procedural' and 'Mesh Filter' output modes. This changes how the mesh data is handed over to Unity for rendering. The 'Mesh Filter' mode simply drags the mesh data back onto the CPU and passes it in to a Mesh Filter component. Procedural mode is waaaay faster - using Unity's DrawProceduralIndirect to keep the data GPU-side. However, you will need a material which is capable of rendering geometry passed in via ComputeBuffers. This project is in URP, which makes it a bit of a pain to hand-write shaders, [and Unity's ShaderGraph doesn't currently support this.](https://forum.unity.com/threads/graphicsbuffer-mesh-vertices-and-compute-shaders.777548/) In my own project, I use Amplify Shader Editor. While I obviously can't include that, I do include the custom node: 'AmplifyNode.hlsl'. Hopefully the feature comes to ShaderGraph soon!

I'll probably want to make these generated meshes interact physically in the future anyway, which will unfortunately require the meshes on the CPU side too.

## Raymarching
If you're familiar with SDFs, you're familiar with raymarching. They very often go hand-in-hand. [Raymarching will also be very familiar to you if you ever go on ShaderToy.](https://www.shadertoy.com/results?query=raymarch) Again I recommend checking out Inigo Quilez for an in-depth explanation, but raymarching is basically an iterative sort of 'pseudo-raytracing' algorithm for rendering complex surfaces like SDFs.

In this project you can use an SDFGroupRaymarcher to visualize your SDFGroup. This component basically just creates a cube mesh and assigns it a special raymarching material. The resulting visual is much more accurate than the isosurface extraction, but it's expensive just to look at: unlike isosurface extraction which is just doing nothing while you're not manipulating the SDFs, raymarching is an active process on your GPU.

The raymarching material is set up to be easy to modify - it's built around this subgraph:

![isomesh5](https://user-images.githubusercontent.com/18707147/115975216-c5da7480-a55a-11eb-9452-18740e81286b.png)

'Hit' is simply a 0/1 int which tells you whether a surface was hit, 'Normal' is that point's surface normal, and that should be enough to set up your shader. I also provide a 'thickness' value to start you on your way to subsurface scattering. Neat! 

It also outputs a UV. UVs are generated procedurally from primitives and for meshes they're sampled. The final UV of each point is a weighted average of the UV of each SDF in the group. Texturing combined shapes can look really funky:

![isomesh6](https://user-images.githubusercontent.com/18707147/115975359-fc64bf00-a55b-11eb-8aa8-b3895448221e.png)

You can also directly visualize the UVs and iteration count.

![isomesh7](https://user-images.githubusercontent.com/18707147/115975420-8745b980-a55c-11eb-9a6f-416848f5cc9e.png)

## Physics

I also include a very fun sample scene showing how you might add physical interaction. Unfortunately, Unity doesn't allow for custom colliders at this time, nor does it allow for non-static concave meshes. Which leaves me pretty limited. However, Unity does allow for convex mesh colliders and even static concave mesh colliders. Creating mesh colliders is very expensive for large meshes though. This led me to experiment with generating very small colliders only around Rigidbodies, at fixed distance intervals.

![blobbyBricks9](https://user-images.githubusercontent.com/18707147/118203358-caa49100-b453-11eb-9d3a-a5af4fff5cca.gif)

It works surprisingly well, even when moving the sdfs around!

## Roadmap and Notes

* ~~I want to be able to add physics to the generated meshes. In theory this should be as simple as adding a MeshCollider and Rigidbody to them, but Unity probably won't play well with these high-poly non-convex meshes, so I may need to split them into many convex meshes.~~
* ~~I intend to add more sdf operations which aren't tied to specific sdf objects, so I can stretch or bend the entire space.
* I'd like to figure out how to get the generated 'UV field' to play nicely with seams on SDFMeshes. Currently I just clamp the interpolated UVs if I detect too big a jump between two neighbouring UV samples.
* None of this stuff is particularly cheap on the GPU. I made no special effort to avoid branching and I could probably use less kernels in the mesh generation process.
* ~~Undo is not fully supported in custom editors yet.
* ~~Some items, especially SDF meshes, don't always cope nicely with all the different transitions Unity goes to, like entering play mode, or recompiling. I've spent a lot of time improving stability in this regard but it's not yet 100%.
* I don't currently use any sort of adaptive octree approach. I consider this a "nice to have."
* I might make a component to automate the "chunking" process, basically just currently positioning the distinct SDFGroupMeshGenerator components, disabling occluded ones, spawning new ones, etc.

![isomesh9](https://user-images.githubusercontent.com/18707147/115975715-03410100-a55f-11eb-8c41-3b983217ba64.gif)


## Known Issues

* Asynchronous mode is not currently recommended, because while it works, the NativeArrays created in the coroutine aren't disposed of when the coroutine is interrupted, for example by going from edit mode to play mode.

## Useful References and Sources

* [Inigo Quilez](https://www.iquilezles.org/)
* [Dual Contouring Tutorial](https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/)
* [Analysis and Acceleration of High QualityIsosurface Contouring](http://www.inf.ufrgs.br/~comba/papers/thesis/diss-leonardo.pdf)
* [Kosmonaut's Signed Distance Field Journey - a fellow SDF mesh creator](https://kosmonautblog.wordpress.com/2017/05/01/signed-distance-field-rendering-journey-pt-1/)
* [DreamCat Games' tutorial on Surface Nets](https://bonsairobo.medium.com/smooth-voxel-mapping-a-technical-deep-dive-on-real-time-surface-nets-and-texturing-ef06d0f8ca14)
* [Local interpolation of surfaces using normal vectors - I use this during the tessellation process to produce smoother geometry](https://stackoverflow.com/questions/25342718/local-interpolation-of-surfaces-using-normal-vectors)
* [Nick's Voxel Blog - good source for learning about implementing the QEF minimizer](http://ngildea.blogspot.com/) [(and their github repo)](https://github.com/nickgildea/qef)
* [MudBun - I came across this tool while already deep in development of this, but it looks awesome and way more clean and professional than this learning exercise.](https://assetstore.unity.com/packages/tools/particles-effects/mudbun-volumetric-vfx-mesh-tool-177891)
