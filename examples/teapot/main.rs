mod mesh;
mod renderer;
mod skybox;

use std::f32::consts::PI;

use mesh::Mesh;
use renderer::{
    make_perspective_proj_matrix, MeshRenderRequest, NormalCompressionType, Renderer,
    ALL_COMPRESSION_TYPES,
};
use skybox::Skybox;
use static_assertions::assert_cfg;

assert_cfg!(
    feature = "half",
    "The \"half\" feature must be enabled for this example to work. Try adding --all-features"
);
assert_cfg!(
    feature = "bytemuck",
    "The \"bytemuck\" feature must be enabled for this example to work. Try adding --all-features"
);

const FRAMEBUFFER_WIDTH: u32 = 1920;
const FRAMEBUFFER_HEIGHT: u32 = 1080;

const NEAR_PLANE_DISTANCE: f32 = 0.1;
const FAR_PLANE_DISTANCE: f32 = 1000.0;
const VERTICAL_FOV: f32 = PI / 4.0;
const ASPECT_RATIO: f32 = FRAMEBUFFER_WIDTH as f32 / FRAMEBUFFER_HEIGHT as f32;

async fn run_example() {
    let renderer = Renderer::new(FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT).await;

    let camera_projection = make_perspective_proj_matrix(
        NEAR_PLANE_DISTANCE,
        FAR_PLANE_DISTANCE,
        VERTICAL_FOV,
        ASPECT_RATIO,
    );

    let camera_transform = glam::Affine3A::from_rotation_translation(
        glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 0.0), PI / 2.0),
        glam::Vec3::new(24.0, 0.0, 0.0),
    );

    let skybox = Skybox::load_from_cubemap_faces(
        &renderer.device,
        &renderer.queue,
        [
            include_bytes!("assets/skybox/posx.jpg"),
            include_bytes!("assets/skybox/negx.jpg"),
            include_bytes!("assets/skybox/posy.jpg"),
            include_bytes!("assets/skybox/negy.jpg"),
            include_bytes!("assets/skybox/posz.jpg"),
            include_bytes!("assets/skybox/negz.jpg"),
        ],
    );

    let teapot_mesh =
        Mesh::load_from_obj_file(&renderer.device, include_bytes!("assets/teapot.obj"));

    for normal_compression_type in ALL_COMPRESSION_TYPES {
        let mesh_render_request = MeshRenderRequest {
            mesh: &teapot_mesh,
            transform: glam::Affine3A::from_rotation_translation(
                glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 0.0), PI / 2.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
            ),
            color: glam::Vec3::new(0.4, 0.2, 0.2),
            normal_compression_type,
        };

        renderer.render(
            camera_transform,
            camera_projection,
            &skybox,
            &[mesh_render_request],
        );

        let framebuffer_image_encoded = renderer.read_framebuffer_to_cpu().await;
        let output_path = format!(
            "teapot_{}.png",
            match normal_compression_type {
                NormalCompressionType::None => "no_packing",
                NormalCompressionType::PackedF32 => "packed_f32",
                NormalCompressionType::PackedF16 => "packed_f16",
                NormalCompressionType::PackedU8 => "packed_u8",
            }
        );
        framebuffer_image_encoded.save(&output_path).unwrap();

        println!("Wrote result to {output_path}");
    }

    println!("Done");
}

fn main() {
    pollster::block_on(run_example());
}
