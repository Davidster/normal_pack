use static_assertions::assert_cfg;

assert_cfg!(feature = "zerocopy", "The \"zerocopy\" feature must be enabled for this example to work. Try adding --features=\"zerocopy\"");

#[cfg(feature = "zerocopy")]
mod example {
    #[repr(C)]
    #[derive(Debug, Copy, Clone, zerocopy::AsBytes, zerocopy:: FromBytes)]
    struct Vertex {
        normal: normal_pack::EncodedUnitVector3F16,
    }

    pub fn run() {
        let normal = [-0.5082557, 0.54751796, 0.6647558];

        let vertex = Vertex {
            normal: normal_pack::EncodedUnitVector3F16::encode(normal),
        };

        let bytes: [u8; 4] = zerocopy::transmute!(vertex);
        let recasted: Vertex = zerocopy::transmute!(bytes);
        let decoded_normal = recasted.normal.decode();

        println!("Bytes: {bytes:?}");
        println!("Recasted: {recasted:?}");
        println!("Decoded normal: {decoded_normal:#?}");
    }
}

fn main() {
    #[cfg(feature = "zerocopy")]
    example::run();
}
