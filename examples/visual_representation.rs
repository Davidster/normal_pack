use std::{
    collections::{hash_map::Entry, HashMap},
    io::{BufReader, Cursor, Read},
};

use static_assertions::assert_cfg;
use wgpu::util::DeviceExt;

assert_cfg!(
    feature = "half",
    "The \"half\" feature must be enabled for this example to work. Try adding --all-features"
);
assert_cfg!(
    feature = "bytemuck",
    "The \"bytemuck\" feature must be enabled for this example to work. Try adding --all-features"
);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal_packed: normal_pack::EncodedUnitVector3,
    pub normal_packed_f16: normal_pack::EncodedUnitVector3F16,
    pub normal_packed_u8: normal_pack::EncodedUnitVector3U8,
    pub padding: [u8; 2],
}

impl ShaderVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // normal
        2 => Float32x2, // tex_coords
        3 => Float32x2, // normal_packed
        4 => Float16x2, // normal_packed_f16
        5 => Uint8x2, // normal_packed_u8
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ShaderVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderCamera {
    view_proj: [[f32; 4]; 4],
    position: [f32; 3],
    padding: f32,
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderObject {
    model_transform: glam::Mat4,
    normal_compression_type: u32,
    padding: [u32; 3],
}

async fn run_example() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .unwrap();

    println!(
        "WGPU device initialized with:\nAdapter: {:?}\nFeatures: {:?}",
        adapter.get_info(),
        device.features()
    );

    let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Blit Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("assets/blit.wgsl").into()),
    });

    let skybox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Skybox Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("assets/skybox.wgsl").into()),
    });

    let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mesh Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("assets/mesh.wgsl").into()),
    });

    let camera_and_object_uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("camera_and_object_uniform_bind_group_layout"),
        });

    let skybox_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // skybox_texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                    },
                    count: None,
                },
                // skybox_sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("skybox_bind_group_layout"),
        });

    let object_material_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("object_material_bind_group_layout"),
        });

    let fragment_shader_color_targets = &[Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        blend: Some(wgpu::BlendState::REPLACE),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mesh Pipeline Layout"),
        bind_group_layouts: &[
            &camera_and_object_uniform_bind_group_layout,
            &object_material_bind_group_layout,
            &skybox_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let mesh_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Mesh Pipeline"),
        layout: Some(&mesh_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &mesh_shader,
            entry_point: "vs_main",
            buffers: &[ShaderVertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &mesh_shader,
            entry_point: "fs_main",
            targets: fragment_shader_color_targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    };

    let mesh_pipeline = device.create_render_pipeline(&mesh_pipeline_descriptor);

    let skybox_pipeline_primitive_state = wgpu::PrimitiveState {
        front_face: wgpu::FrontFace::Cw,
        ..Default::default()
    };
    let skybox_depth_stencil_state = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::GreaterEqual,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });
    let skybox_render_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Render Pipeline Layout"),
            bind_group_layouts: &[
                &camera_and_object_uniform_bind_group_layout,
                &skybox_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let skybox_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Skybox Render Pipeline"),
        layout: Some(&skybox_render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &skybox_shader,
            entry_point: "vs_main",
            buffers: &[ShaderVertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &skybox_shader,
            entry_point: "fs_main",
            targets: fragment_shader_color_targets,
            compilation_options: Default::default(),
        }),
        primitive: skybox_pipeline_primitive_state,
        depth_stencil: skybox_depth_stencil_state,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    };
    let skybox_pipeline = device.create_render_pipeline(&skybox_pipeline_descriptor);

    let image_dimensions = wgpu::Extent3d {
        width: 1920,
        height: 1080,
        depth_or_array_layers: 1,
    };

    let framebuffer = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Framebuffer"),
        size: image_dimensions,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let framebuffer_view = framebuffer.create_view(&Default::default());

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth texture"),
        size: image_dimensions,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_texture_view = depth_texture.create_view(&Default::default());

    // TODO:
    // let skybox_bind_group =

    let camera_projection_matrix = make_perspective_proj_matrix(
        0.1,
        1000.0,
        std::f32::consts::PI / 4.0,
        image_dimensions.width as f32 / image_dimensions.height as f32,
        true,
    );

    let camera_transform = glam::Affine3A::from_rotation_translation(
        glam::Quat::IDENTITY,
        glam::Vec3::new(0.0, 0.0, 0.0),
    );
    let camera_rotation_only_view_matrix =
        glam::Mat4::from_mat3a(camera_transform.matrix3.inverse());
    let camera_view_matrix = glam::Mat4::from(camera_transform.inverse());

    let main_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: &bytemuck::cast_slice(&[ShaderCamera {
            view_proj: (camera_projection_matrix * camera_view_matrix).to_cols_array_2d(),
            position: [
                camera_transform.translation.x,
                camera_transform.translation.y,
                camera_transform.translation.z,
            ],
            padding: Default::default(),
        }]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let skybox_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Skybox Camera Buffer"),
        contents: &bytemuck::cast_slice(&[ShaderCamera {
            view_proj: (camera_projection_matrix * camera_rotation_only_view_matrix)
                .to_cols_array_2d(),
            position: [
                camera_transform.translation.x,
                camera_transform.translation.y,
                camera_transform.translation.z,
            ],
            padding: Default::default(),
        }]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let dummy_object_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dummy Object Buffer"),
        contents: &bytemuck::cast_slice(&[ShaderObject::default()]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let skybox_camera_and_dummy_object_bind_group =
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &camera_and_object_uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        skybox_camera_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        dummy_object_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

    let teapot_mesh = load_mesh(include_bytes!("assets/teapot.obj"));

    let teapot_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Teapot Vertex Buffer"),
        contents: &bytemuck::cast_slice(&teapot_mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let teapot_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Teapot Index Buffer"),
        contents: &bytemuck::cast_slice(&teapot_mesh.indices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let cube_mesh = load_mesh(include_bytes!("assets/cube.obj"));

    dbg!(&cube_mesh.vertices);

    let cube_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cube Vertex Buffer"),
        contents: &bytemuck::cast_slice(&cube_mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let cube_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cube Index Buffer"),
        contents: &bytemuck::cast_slice(&cube_mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let (skybox_cubemap_raw, skybox_cubemap_size) = load_cubemap(
        include_bytes!("assets/skybox/pos_x.png"),
        include_bytes!("assets/skybox/neg_x.png"),
        include_bytes!("assets/skybox/pos_y.png"),
        include_bytes!("assets/skybox/neg_y.png"),
        include_bytes!("assets/skybox/pos_z.png"),
        include_bytes!("assets/skybox/neg_z.png"),
    );

    let skybox_texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("Skybox"),
            size: skybox_cubemap_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &skybox_cubemap_raw,
    );
    let skybox_texture_view = skybox_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    let skybox_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let skybox_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &skybox_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&skybox_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&skybox_sampler),
            },
        ],
    });

    for _ in 0..20 {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Skybox render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &framebuffer_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            render_pass.set_pipeline(&skybox_pipeline);
            render_pass.set_bind_group(0, &skybox_camera_and_dummy_object_bind_group, &[]);
            render_pass.set_bind_group(1, &skybox_bind_group, &[]);

            render_pass.set_vertex_buffer(0, cube_vertex_buffer.slice(..));
            render_pass.set_index_buffer(cube_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..(cube_mesh.indices.len() as u32), 0, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        println!("Sleeping...");
        std::thread::sleep(std::time::Duration::from_secs_f32(0.1));
    }

    let mut framebuffer_bytes = texture_to_bytes(&device, &queue, &framebuffer).await;
    let framebuffer_image_encoded = image::RgbaImage::from_raw(
        framebuffer.size().width,
        framebuffer.size().height,
        framebuffer_bytes.pop().unwrap(),
    )
    .unwrap();

    framebuffer_image_encoded.save("out.png").unwrap();
}

pub fn make_perspective_proj_matrix(
    near_plane_distance: f32,
    far_plane_distance: f32,
    fov_y: f32,
    aspect_ratio: f32,
    reverse_z: bool,
) -> glam::Mat4 {
    let n = near_plane_distance;
    let f = far_plane_distance;
    let cot = 1.0 / (fov_y / 2.0).tan();
    let ar = aspect_ratio;
    #[rustfmt::skip]
  let persp_matrix = glam::Mat4::from_cols_array(&[
      cot/ar, 0.0, 0.0,     0.0,
      0.0,    cot, 0.0,     0.0,
      0.0,    0.0, f/(n-f), n*f/(n-f),
      0.0,    0.0, -1.0,     0.0,
  ]).transpose();
    if !reverse_z {
        persp_matrix
    } else {
        #[rustfmt::skip]
      let reverse_z = glam::Mat4::from_cols_array(&[
          1.0, 0.0, 0.0,  0.0,
          0.0, 1.0, 0.0,  0.0,
          0.0, 0.0, -1.0, 1.0,
          0.0, 0.0, 0.0,  1.0,
      ]).transpose();
        reverse_z * persp_matrix
    }
}

pub struct BasicMesh {
    pub vertices: Vec<ShaderVertex>,
    pub indices: Vec<u16>,
}

fn load_mesh(obj_file_bytes: &[u8]) -> BasicMesh {
    let obj = obj::raw::parse_obj(BufReader::new(Cursor::new(obj_file_bytes))).unwrap();

    let mut triangles: Vec<[(usize, usize, usize); 3]> = vec![];

    for polygon in obj.polygons.iter() {
        match polygon {
            obj::raw::object::Polygon::PTN(points) => {
                if points.len() < 3 {
                    panic!("BasicMesh requires that all polygons have at least 3 vertices");
                }

                let last_elem = points.last().unwrap();

                triangles.extend(
                    points[..points.len() - 1]
                        .iter()
                        .zip(points[1..points.len() - 1].iter())
                        .map(|(&x, &y)| [*last_elem, x, y]),
                );
            }
            // obj::raw::object::Polygon::PN(points) => {
            //     if points.len() < 3 {
            //         panic!("BasicMesh requires that all polygons have at least 3 vertices");
            //     }

            //     let points: Vec<_> = points
            //         .iter()
            //         .map(|point| (point.0, point.1, 0usize))
            //         .collect();

            //     let last_elem = points.last().unwrap();

            //     triangles.extend(
            //         points[..points.len() - 1]
            //             .iter()
            //             .zip(points[1..points.len() - 1].iter())
            //             .map(|(x, y)| [*last_elem, *x, *y]),
            //     );
            // }
            _ => {
                panic!("BasicMesh requires that all points have a position and normal");
            }
        }
    }

    let mut composite_index_map: HashMap<(usize, usize, usize), ShaderVertex> = HashMap::new();
    triangles.iter().for_each(|triangle_vertices| {
        triangle_vertices.iter().for_each(|vti| {
            let pos = obj.positions[vti.0];
            let normal = obj.normals[vti.2];
            let normal = [normal.0, normal.1, normal.2];
            let uv = obj.tex_coords[vti.1];

            if let Entry::Vacant(vacant_entry) = composite_index_map.entry(*vti) {
                vacant_entry.insert(ShaderVertex {
                    position: [pos.0, pos.1, pos.2],
                    normal,
                    tex_coords: [uv.0, 1.0 - uv.1], // convert uv format into 0->1 range
                    normal_packed: normal_pack::EncodedUnitVector3::new(normal),
                    normal_packed_f16: normal_pack::EncodedUnitVector3F16::new(normal),
                    normal_packed_u8: normal_pack::EncodedUnitVector3U8::new(normal),
                    padding: Default::default(),
                });
            }
        });
    });
    let mut index_map: HashMap<(usize, usize, usize), usize> = HashMap::new();
    let mut vertices: Vec<ShaderVertex> = Vec::new();
    composite_index_map
        .iter()
        .enumerate()
        .for_each(|(i, (key, vertex))| {
            index_map.insert(*key, i);
            vertices.push(*vertex);
        });
    let indices: Vec<_> = triangles
        .iter()
        .flat_map(|points| {
            points
                .iter()
                .flat_map(|vti| {
                    let key = (vti.0, vti.2, vti.1);
                    index_map.get(&key).map(|final_index| *final_index as u16)
                })
                .collect::<Vec<u16>>()
        })
        .collect();
    BasicMesh { vertices, indices }
}

fn load_cubemap(
    pos_x_bytes: &[u8],
    neg_x_bytes: &[u8],
    pos_y_bytes: &[u8],
    neg_y_bytes: &[u8],
    pos_z_bytes: &[u8],
    neg_z_bytes: &[u8],
) -> (Vec<u8>, wgpu::Extent3d) {
    let images = [
        pos_x_bytes,
        neg_x_bytes,
        pos_y_bytes,
        neg_y_bytes,
        pos_z_bytes,
        neg_z_bytes,
    ];

    let mut all_bytes: Vec<u8> = Vec::new();

    let mut height = 0;
    let mut width = 0;

    for image_encoded in images {
        let image = image::load_from_memory(image_encoded).unwrap();
        height = image.height();
        width = image.width();

        for byte in image.as_rgba8().unwrap().as_raw() {
            all_bytes.push(*byte);
        }
    }

    (
        all_bytes,
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 6,
        },
    )
}

async fn texture_to_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Vec<Vec<u8>> {
    let mut result = vec![];
    for depth in 0..texture.size().depth_or_array_layers {
        let mut depth_level_bytes = vec![];
        for mip_level in 0..texture.mip_level_count() {
            let width = texture.size().width >> mip_level;
            let height = texture.size().height >> mip_level;
            let unpadded_bytes_per_row = unpadded_bytes_per_row(texture, Some(mip_level));
            let padded_bytes_per_row = padded_bytes_per_row(texture, Some(mip_level));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("to_bytes encoder"),
            });

            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                size: (padded_bytes_per_row * height) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                label: Some("to_bytes output buffer"),
                mapped_at_creation: false,
            });

            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    aspect: wgpu::TextureAspect::All,
                    texture: &texture,
                    mip_level,
                    origin: wgpu::Origin3d {
                        z: depth,
                        ..wgpu::Origin3d::ZERO
                    },
                },
                wgpu::ImageCopyBuffer {
                    buffer: &output_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            queue.submit(Some(encoder.finish()));

            {
                let buffer_slice = output_buffer.slice(..);

                let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
                device.poll(wgpu::Maintain::Wait);
                rx.receive().await.unwrap().unwrap();

                let data = &buffer_slice.get_mapped_range();
                depth_level_bytes.reserve((unpadded_bytes_per_row * height) as usize);
                for i in 0..height {
                    let start_index = (i * padded_bytes_per_row) as usize;
                    let end_index = start_index + unpadded_bytes_per_row as usize;
                    depth_level_bytes.extend_from_slice(&data[start_index..end_index]);
                }
            }
            output_buffer.unmap();
        }
        result.push(depth_level_bytes);
    }

    result
}

pub fn unpadded_bytes_per_row(texture: &wgpu::Texture, mip_level: Option<u32>) -> u32 {
    (texture.size().width >> mip_level.unwrap_or(0))
        * texture.format().block_copy_size(None).expect(
            "This function was not meant to be called with a Depth/Stencil/Planar texture format",
        )
}

pub fn padded_bytes_per_row(texture: &wgpu::Texture, mip_level: Option<u32>) -> u32 {
    let unpadded_bytes_per_row = unpadded_bytes_per_row(texture, mip_level);
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
    unpadded_bytes_per_row + padded_bytes_per_row_padding
}

fn main() {
    pollster::block_on(run_example());
}
