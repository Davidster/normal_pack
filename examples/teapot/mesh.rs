use std::{
    collections::{hash_map::Entry, HashMap},
    io::{BufReader, Cursor},
};

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderVertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub tex_coords: glam::Vec2,
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

    pub fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ShaderVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct Mesh {
    pub _vertices: Vec<ShaderVertex>,
    pub indices: Vec<u16>,

    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
}

impl Mesh {
    pub fn load_from_obj_file(device: &wgpu::Device, obj_file_bytes: &[u8]) -> Self {
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
                _ => {
                    panic!("BasicMesh requires that all points have a position and normal");
                }
            }
        }

        let mut composite_index_map: HashMap<(usize, usize, usize), ShaderVertex> = HashMap::new();
        triangles.iter().for_each(|triangle_vertices| {
            triangle_vertices.iter().for_each(|vti| {
                let key = (vti.0, vti.2, vti.1);
                let pos = obj.positions[vti.0];
                let normal = obj.normals[vti.2];
                let normal = [normal.0, normal.1, normal.2];
                let uv = obj.tex_coords[vti.1];

                if let Entry::Vacant(vacant_entry) = composite_index_map.entry(key) {
                    vacant_entry.insert(ShaderVertex {
                        position: glam::Vec3::from([pos.0, pos.1, pos.2]),
                        normal: glam::Vec3::from(normal),
                        tex_coords: glam::Vec2::from([uv.0, 1.0 - uv.1]), // convert uv format into 0->1 range
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

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            _vertices: vertices,
            indices,
            vertex_buffer,
            index_buffer,
        }
    }
}
