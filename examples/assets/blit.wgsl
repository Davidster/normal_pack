

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

// should be called with 3 vertex indices: 0, 1, 2
// draws one large triangle over the clip space like this:
// (the asterisks represent the clip space bounds)
//-1,1           1,1
// ---------------------------------
// |              *              .
// |              *           .
// |              *        .
// |              *      .
// |              *    . 
// |              * .
// |***************
// |            . 1,-1 
// |          .
// |       .
// |     .
// |   .
// |.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    out.position = vec4<f32>(
        tc.x * 2.0 - 1.0,
        1.0 - tc.y * 2.0,
        0.0,
        1.0
    );
    out.tex_coords = tc;
    return out;
}

@group(0) @binding(0)
var texture_1: texture_2d<f32>;
@group(0) @binding(1)
var sampler_1: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(texture_1, sampler_1, in.tex_coords);
}