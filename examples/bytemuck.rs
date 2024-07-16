use static_assertions::assert_cfg;

assert_cfg!(feature = "bytemuck", "The \"bytemuck\" feature must be enabled for this example to work. Try adding --features=\"bytemuck\"");

#[cfg(feature = "bytemuck")]
mod example {
    use bytemuck::{Pod, Zeroable};
    use normal_pack::EncodedUnitVector3F16;

    #[repr(C)]
    #[derive(Debug, Pod, Zeroable, Copy, Clone)]
    struct Vertex {
        normal: EncodedUnitVector3F16,
    }

    pub fn run() {
        let normal = [-0.5082557, 0.54751796, 0.6647558];

        let vertex = Vertex {
            normal: EncodedUnitVector3F16::new(normal),
        };

        let bytes: [u8; 4] = bytemuck::cast(vertex);
        let recasted: Vertex = bytemuck::cast(bytes);
        let decoded_normal = recasted.normal.to_array();

        println!("Bytes: {bytes:?}");
        println!("Recasted: {recasted:?}");
        println!("Decoded normal: {decoded_normal:#?}");
    }
}

fn main() {
    #[cfg(feature = "bytemuck")]
    example::run();
}
