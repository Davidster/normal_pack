fn oct_wrap(v: vec2<f32>) -> vec2<f32> {
    return ( 1.0 - abs( v.yx ) ) * ( select(vec2(-1.0), vec2(1.0), v.xy >= vec2(0.0)) );
}

// TODO: proofread, and consider using twitter technique
fn oct_decode_unit_vector_float(in: vec2<f32>) -> vec3<f32> {
    var encN = in;

    // TODO: is this needed?
    // encN = encN * 2.0 - 1.0;

    var n: vec3<f32>;
    n.z = 1.0 - abs( encN.x ) - abs( encN.y );
    n = vec3(select(oct_wrap( encN.xy ), encN.xy, n.z >= 0.0), n.z);
    n = normalize( n );
    return n;
}

fn oct_decode_unit_vector_u8(in: vec2<u32>) -> vec3<f32> {
    return oct_decode_unit_vector_float((vec2<f32>(in) / 255.0) * vec2(2.0) - vec2(1.0));
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
}

struct ObjectUniform {
    model_transform_0: vec4<f32>,
    model_transform_1: vec4<f32>,
    model_transform_2: vec4<f32>,
    model_transform_3: vec4<f32>,
    normal_compression_type: vec4<u32>, // last three elements are ignored, they are just padding
}

@group(0) @binding(0)
var<uniform> camera_uniform: CameraUniform;

@group(0) @binding(1)
var<uniform> object_uniform: ObjectUniform;

struct VertexInput {
    @location(0) object_position: vec3<f32>,
    @location(1) object_normal: vec3<f32>,
    @location(2) object_tex_coords: vec2<f32>,
    @location(3) object_normal_packed: vec2<f32>,
    @location(4) object_normal_packed_f16: vec2<f32>,
    @location(5) object_normal_packed_u8: vec2<u32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    vshader_input: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {

    let model_transform = mat4x4<f32>(
        object_uniform.model_transform_0,
        object_uniform.model_transform_1,
        object_uniform.model_transform_2,
        object_uniform.model_transform_3,
    );

    var object_normal: vec3<f32>;
    let normal_compression_type = object_uniform.normal_compression_type.x;
    if normal_compression_type == 0 {
      object_normal = vshader_input.object_normal;
    } else if normal_compression_type == 1 {
      object_normal = oct_decode_unit_vector_float(vshader_input.object_normal_packed);
    } else if normal_compression_type == 2 {
      object_normal = oct_decode_unit_vector_float(vshader_input.object_normal_packed_f16);
    } else if normal_compression_type == 2 {
      object_normal = oct_decode_unit_vector_u8(vshader_input.object_normal_packed_u8);
    }

    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let world_position = model_transform * object_position;
    let clip_position = camera_uniform.view_proj * model_transform * object_position;
    let world_normal = normalize((model_transform * vec4<f32>(object_normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_position = clip_position;
    out.world_position = world_position.xyz;
    out.world_normal = world_normal;
    out.tex_coords = vshader_input.object_tex_coords;

    return out;
}

@group(1) @binding(0)
var diffuse_texture: texture_2d<f32>;
@group(1) @binding(1)
var diffuse_sampler: sampler;

@group(2) @binding(0)
var skybox_texture: texture_cube<f32>;
@group(2) @binding(1)
var skybox_sampler: sampler;

fn fresnel_direct(
    cos_theta: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_ambient(
    cos_theta: f32,
    f0: vec3<f32>,
    a: f32,
) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - a), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    // return f0 + (max(vec3<f32>(1.0 - a), f0) - f0) * pow(1.0 - h_dot_v, 5.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>  {
    let light_position = vec3(1.0, 1.0, 1.0);
    let f0 = vec3(0.04);
    let roughness = 0.1;
    let exposure = 5.0;

    let to_viewer_vec = normalize(camera_uniform.position.xyz - in.world_position);
    let to_light_vec = normalize(in.world_position - light_position);
    let reflection_vec = reflect(-to_viewer_vec, normalize(in.world_normal));

    let base_color = textureSample(
        diffuse_texture,
        diffuse_sampler,
        in.tex_coords
    ).rgb;

    let skybox_color = textureSample(
        skybox_texture,
        skybox_sampler,
        world_normal_to_cubemap_vec(reflection_vec)
    ).rgb;

    // environment lighting / reflection
    let n_dot_v = max(dot(in.world_normal, to_viewer_vec), 0.0);
    let fresnel_ambient = fresnel_ambient(n_dot_v, f0, roughness);
    let ambient_lighting = skybox_color * fresnel_ambient + (1.0 - fresnel_ambient) * base_color;
    
    // direct lighting
    let incident_angle_factor = max(dot(in.world_normal, to_light_vec), 0.0);
    let halfway_vec = normalize(to_viewer_vec + to_light_vec);
    let h_dot_v = max(dot(halfway_vec, to_viewer_vec), 0.0);
    let fresnel_direct = fresnel_direct(h_dot_v, f0);
    let direct_lighting = base_color * incident_angle_factor * fresnel_direct;
    
    let combined_lighting_hdr = ambient_lighting + direct_lighting;
    let combined_lighting_ldr = 1.0 - exp(-combined_lighting_hdr * exposure);

    return vec4(combined_lighting_ldr, 1.0);
}

fn world_normal_to_cubemap_vec(world_pos: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(-world_pos.x, world_pos.y, world_pos.z);
}