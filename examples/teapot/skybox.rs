use wgpu::util::DeviceExt;

pub struct Skybox {
    pub texture: wgpu::Texture,
    pub sampler: wgpu::Sampler,
}

impl Skybox {
    pub fn load_from_cubemap_faces(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        faces: [&[u8]; 6],
    ) -> Self {
        let mut combined_image_bytes: Vec<u8> = Vec::new();

        let mut height = 0;
        let mut width = 0;

        for image_encoded in faces {
            let image = image::load_from_memory(image_encoded).unwrap();
            height = image.height();
            width = image.width();

            for byte in image.to_rgba8().as_raw() {
                combined_image_bytes.push(*byte);
            }
        }

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 6,
        };

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Skybox"),
                size,
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
            &combined_image_bytes,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self { texture, sampler }
    }
}
