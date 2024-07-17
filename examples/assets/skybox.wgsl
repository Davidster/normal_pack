struct CameraUniform {
    rotation_only_view_proj: mat4x4<f32>,
    position: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera_uniform: CameraUniform;

struct VertexInput {
    @location(0) object_position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

@vertex
fn vs_main(
    vshader_input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let clip_position = camera_uniform.rotation_only_view_proj * vec4<f32>(vshader_input.object_position, 1.0);
    out.clip_position = vec4<f32>(clip_position.x, clip_position.y, 0.0, clip_position.w);
    out.world_position = vshader_input.object_position;
    return out;
}

@group(1) @binding(0)
var skybox_texture: texture_cube<f32>;
@group(1) @binding(1)
var skybox_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let col = textureSample(skybox_texture, skybox_sampler, world_normal_to_cubemap_vec(in.world_position));
    return vec4<f32>(col.xyz, 1.0);
    // return vec4<f32>(in.world_position, 1.0);
    // return in.clip_position / in.clip_position.w;
}

fn world_normal_to_cubemap_vec(world_pos: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(-world_pos.x, world_pos.y, world_pos.z);
}