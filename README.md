## Compresses normal vectors (or any 3D unit vector) using [Octahedron encoding](https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/).

[![Crates.io](https://img.shields.io/crates/v/normal_pack.svg)](https://crates.io/crates/normal_pack/) [![Documentation](https://docs.rs/normal_pack/badge.svg)](https://docs.rs/normal_pack/) ![Crates.io](https://img.shields.io/crates/l/normal_pack)

This lossy compression scheme is able to achieve a compression ratio as high as 6:1 with an average error rate of less than 1 degree,
depending on which representation is chosen.

#### Example:

```
let normal = [-0.5082557, 0.54751796, 0.6647558];

let encoded = normal_pack::EncodedUnitVector3U8::encode(normal);
let decoded = encoded.decode();

assert_eq!(decoded, [-0.52032965, 0.5473598, 0.6554802]);

```

#### Why compress my normals?

It is common for 3D renderers to be bottlenecked by memory bandwidth, such as when loading normals from VRAM for high-poly meshes to supply to your vertex shader.
A smaller memory footprint for your normals corresponds to memory bandwidth savings and higher FPS in such scenarios.

#### How bad is 1 degree of error?

The `teapot` example generates a reference visual and contains the wgsl code required to decode the vector in a shader.

##### Standard [f32; 3] representation
![teapot_packed_u8](https://github.com/user-attachments/assets/b16818d0-8020-477a-b6ec-99966eb1ae85)

##### Packed into a [u8; 2]
![teapot_no_packing](https://github.com/user-attachments/assets/6e6ab8ad-37da-4be0-b8ef-e17c0ae9614f)

##### As a video

![normal_pack_error](https://github.com/user-attachments/assets/c4070012-9b8d-4573-bc18-f02f54101c67)

*The skybox used in the example is the work of Emil Persson, aka Humus. [http://www.humus.name](http://www.humus.name)*
