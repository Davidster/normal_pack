fn oct_wrap(v: vec2<f32>) -> vec2<f32> {
    return ( 1.0 - abs( v.yx ) ) * ( select(vec2(-1.0), vec2(1.0), v.xy >= vec2(0.0)) );
}

// TODO: proofread, and consider using twitter technique
fn oct_decode_unit_vector_float(in: vec2<f32>) -> vec3<f32> {
    var encN = in;

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
    color: vec3<f32>,
    normal_compression_type: u32,
}

@group(0) @binding(0)
var<uniform> camera_uniform: CameraUniform;

@group(0) @binding(1)
var<uniform> object_uniform: ObjectUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) normal_packed: vec2<f32>,
    @location(4) normal_packed_f16: vec2<f32>,
    @location(5) normal_packed_u8: vec2<u32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) object_color: vec3<f32>,
    @location(4) object_center: vec3<f32>,
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

    var normal: vec3<f32>;
    let normal_compression_type = object_uniform.normal_compression_type;
    if normal_compression_type == 0 {
      normal = vshader_input.normal;
    } else if normal_compression_type == 1 {
      normal = oct_decode_unit_vector_float(vshader_input.normal_packed);
    } else if normal_compression_type == 2 {
      normal = oct_decode_unit_vector_float(vshader_input.normal_packed_f16);
    } else if normal_compression_type == 3 {
      normal = oct_decode_unit_vector_u8(vshader_input.normal_packed_u8);
    }

    let position = vec4<f32>(vshader_input.position, 1.0);
    let world_position = model_transform * position;
    let clip_position = camera_uniform.view_proj * model_transform * position;
    let world_normal = normalize((model_transform * vec4<f32>(normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_position = clip_position;
    out.world_position = world_position.xyz;
    out.world_normal = world_normal;
    out.tex_coords = vshader_input.tex_coords;
    out.object_color = object_uniform.color;
    out.object_center = object_uniform.model_transform_3.xyz;

    return out;
}

@group(1) @binding(0)
var skybox_texture: texture_cube<f32>;
@group(1) @binding(1)
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
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>  {
    let light_position = vec3(-10.0, 5.0, -30.0);
    let light_intensity = 1.0;
    let f0 = vec3(0.2);
    let roughness = 0.1;

    let to_viewer_vec = normalize(camera_uniform.position.xyz - in.world_position);
    let to_light_vec = normalize(in.world_position - light_position);
    let reflection_vec = reflect(-to_viewer_vec, in.world_normal);

    let base_color = in.object_color;

    let skybox_texture_color = textureSample(
        skybox_texture,
        skybox_sampler,
        world_normal_to_cubemap_vec(reflection_vec)
    ).rgb;
    let skybox_color = skybox_texture_color;

    // environment lighting / reflection
    let n_dot_v = max(dot(in.world_normal, to_viewer_vec), 0.0);
    let fresnel_ambient = fresnel_ambient(n_dot_v, f0, roughness);
    let ambient_lighting = skybox_color * fresnel_ambient + (1.0 - fresnel_ambient) * base_color;
    
    // direct lighting
    let incident_angle_factor = max(dot(in.world_normal, to_light_vec), 0.0);
    let halfway_vec = normalize(to_viewer_vec + to_light_vec);
    let h_dot_v = max(dot(halfway_vec, to_viewer_vec), 0.0);
    let fresnel_direct = fresnel_direct(h_dot_v, f0);
    let direct_lighting = light_intensity * base_color * incident_angle_factor * fresnel_direct;
    
    let combined_lighting_hdr = ambient_lighting + direct_lighting;

    // blend between the normal color and the lit color about the plane defined by the diagonal and point of intersection
    let diagonal = normalize(vec3(0.0, 1.0, 1.0));
    let intersection = in.object_center - diagonal * 3.0;
    let blended_area_width = 0.125;
    let t = clamp(dot(normalize(in.world_position - intersection), diagonal) / blended_area_width, 0.0, 2.0);
    let t_smooth = smoothstep(0.0, 1.0, 0.5 * t);

    return vec4(mix(in.world_normal, combined_lighting_hdr, t_smooth), 1.0);
}

fn world_normal_to_cubemap_vec(world_pos: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(-world_pos.x, world_pos.y, world_pos.z);
}