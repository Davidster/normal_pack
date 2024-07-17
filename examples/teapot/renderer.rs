use wgpu::util::DeviceExt;

use crate::{
    mesh::{Mesh, ShaderVertex},
    skybox::Skybox,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderCamera {
    view_proj: glam::Mat4,
    position: glam::Vec3,
    padding: f32,
}

#[repr(C)]
#[derive(
    Default, Debug, Copy, Clone, bytemuck::CheckedBitPattern, bytemuck::NoUninit, bytemuck::Zeroable,
)]
struct ShaderObject {
    model_transform: glam::Mat4,
    color: glam::Vec3,
    normal_compression_type: NormalCompressionType,
}

#[repr(u32)]
#[derive(
    Default, Debug, Copy, Clone, bytemuck::CheckedBitPattern, bytemuck::NoUninit, bytemuck::Zeroable,
)]
pub enum NormalCompressionType {
    #[default]
    None,
    PackedF32,
    PackedF16,
    PackedU8,
}

pub const ALL_COMPRESSION_TYPES: [NormalCompressionType; 4] = [
    NormalCompressionType::None,
    NormalCompressionType::PackedF32,
    NormalCompressionType::PackedF16,
    NormalCompressionType::PackedU8,
];

const MULTISAMPLE_COUNT: u32 = 4;

pub struct MeshRenderRequest<'a> {
    pub mesh: &'a Mesh,
    pub transform: glam::Affine3A,
    pub color: glam::Vec3,
    pub normal_compression_type: NormalCompressionType,
}

pub struct Renderer {
    pub _instance: wgpu::Instance,
    pub _adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub camera_and_object_uniform_bind_group_layout: wgpu::BindGroupLayout,
    pub skybox_bind_group_layout: wgpu::BindGroupLayout,

    pub mesh_pipeline: wgpu::RenderPipeline,
    pub skybox_pipeline: wgpu::RenderPipeline,

    pub multisample_framebuffer: wgpu::Texture,
    pub framebuffer: wgpu::Texture,
    pub depth_texture: wgpu::Texture,

    // used for rendering the skybox
    pub cube: Mesh,
}

impl Renderer {
    pub async fn new(framebuffer_width: u32, framebuffer_height: u32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("skybox_bind_group_layout"),
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
                buffers: &[ShaderVertex::layout()],
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
                count: MULTISAMPLE_COUNT,
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
                buffers: &[ShaderVertex::layout()],
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
            multisample: wgpu::MultisampleState {
                count: MULTISAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };
        let skybox_pipeline = device.create_render_pipeline(&skybox_pipeline_descriptor);

        let image_dimensions = wgpu::Extent3d {
            width: framebuffer_width,
            height: framebuffer_height,
            depth_or_array_layers: 1,
        };

        let multisample_framebuffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Multisample Framebuffer"),
            size: image_dimensions,
            mip_level_count: 1,
            sample_count: MULTISAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

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

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth texture"),
            size: image_dimensions,
            mip_level_count: 1,
            sample_count: MULTISAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let cube = Mesh::load_from_obj_file(&device, include_bytes!("assets/cube.obj"));

        Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,

            camera_and_object_uniform_bind_group_layout,
            skybox_bind_group_layout,

            mesh_pipeline,
            skybox_pipeline,

            multisample_framebuffer,
            framebuffer,
            depth_texture,

            cube,
        }
    }

    pub fn render(
        &self,
        camera_transform: glam::Affine3A,
        camera_projection: glam::Mat4,
        skybox: &Skybox,
        mesh_render_requests: &[MeshRenderRequest],
    ) {
        let camera_rotation_only_view = glam::Mat4::from_mat3a(camera_transform.matrix3.inverse());
        let camera_view = glam::Mat4::from(camera_transform.inverse());

        let skybox_camera_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Skybox Camera Buffer"),
                    contents: bytemuck::cast_slice(&[ShaderCamera {
                        view_proj: camera_projection * camera_rotation_only_view,
                        position: camera_transform.translation.into(),
                        padding: Default::default(),
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let dummy_object_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dummy Object Buffer"),
                    contents: bytemuck::cast_slice(&[ShaderObject::default()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let skybox_camera_and_dummy_object_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.camera_and_object_uniform_bind_group_layout,
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

        let skybox_texture_view = skybox.texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let skybox_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.skybox_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&skybox_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&skybox.sampler),
                },
            ],
        });

        let main_camera_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Buffer"),
                    contents: bytemuck::cast_slice(&[ShaderCamera {
                        view_proj: camera_projection * camera_view,
                        position: camera_transform.translation.into(),
                        padding: Default::default(),
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        self.device.start_capture();

        let ms_framebuffer_view = self
            .multisample_framebuffer
            .create_view(&Default::default());
        let framebuffer_view = self.framebuffer.create_view(&Default::default());
        let depth_texture_view = self.depth_texture.create_view(&Default::default());

        let mesh_buffers: Vec<_> = mesh_render_requests
            .iter()
            .map(|mesh| {
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&[ShaderObject {
                            model_transform: mesh.transform.into(),
                            color: mesh.color.powf(2.2),
                            normal_compression_type: mesh.normal_compression_type,
                        }]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.camera_and_object_uniform_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                main_camera_buffer.as_entire_buffer_binding(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(
                                buffer.as_entire_buffer_binding(),
                            ),
                        },
                    ],
                });

                (buffer, bind_group)
            })
            .collect();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Skybox render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if MULTISAMPLE_COUNT > 1 {
                        &ms_framebuffer_view
                    } else {
                        &framebuffer_view
                    },
                    resolve_target: (MULTISAMPLE_COUNT > 1).then_some(&framebuffer_view),
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

            render_pass.set_pipeline(&self.skybox_pipeline);
            render_pass.set_bind_group(0, &skybox_camera_and_dummy_object_bind_group, &[]);
            render_pass.set_bind_group(1, &skybox_bind_group, &[]);

            render_pass.set_vertex_buffer(0, self.cube.vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(self.cube.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..(self.cube.indices.len() as u32), 0, 0..1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if MULTISAMPLE_COUNT > 1 {
                        &ms_framebuffer_view
                    } else {
                        &framebuffer_view
                    },
                    resolve_target: (MULTISAMPLE_COUNT > 1).then_some(&framebuffer_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
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

            render_pass.set_pipeline(&self.mesh_pipeline);
            render_pass.set_bind_group(1, &skybox_bind_group, &[]);

            for (i, mesh_render_request) in mesh_render_requests.iter().enumerate() {
                render_pass.set_bind_group(0, &mesh_buffers[i].1, &[]);

                render_pass.set_vertex_buffer(0, mesh_render_request.mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    mesh_render_request.mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.draw_indexed(
                    0..(mesh_render_request.mesh.indices.len() as u32),
                    0,
                    0..1,
                );
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.stop_capture();
    }

    pub async fn read_framebuffer_to_cpu(&self) -> image::RgbaImage {
        let mut framebuffer_bytes =
            texture_to_bytes(&self.device, &self.queue, &self.framebuffer).await;
        image::RgbaImage::from_raw(
            self.framebuffer.size().width,
            self.framebuffer.size().height,
            framebuffer_bytes.pop().unwrap(),
        )
        .unwrap()
    }
}

fn unpadded_bytes_per_row(texture: &wgpu::Texture, mip_level: Option<u32>) -> u32 {
    (texture.size().width >> mip_level.unwrap_or(0))
        * texture.format().block_copy_size(None).expect(
            "This function was not meant to be called with a Depth/Stencil/Planar texture format",
        )
}

fn padded_bytes_per_row(texture: &wgpu::Texture, mip_level: Option<u32>) -> u32 {
    let unpadded_bytes_per_row = unpadded_bytes_per_row(texture, mip_level);
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
    unpadded_bytes_per_row + padded_bytes_per_row_padding
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
                    texture,
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

pub fn make_perspective_proj_matrix(
    near_plane_distance: f32,
    far_plane_distance: f32,
    fov_y: f32,
    aspect_ratio: f32,
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
    #[rustfmt::skip]
    let reverse_z = glam::Mat4::from_cols_array(&[
        1.0, 0.0, 0.0,  0.0,
        0.0, 1.0, 0.0,  0.0,
        0.0, 0.0, -1.0, 1.0,
        0.0, 0.0, 0.0,  1.0,
    ]).transpose();
    reverse_z * persp_matrix
}
