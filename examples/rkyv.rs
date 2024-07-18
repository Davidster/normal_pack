use static_assertions::assert_cfg;

assert_cfg!(
    feature = "rkyv",
    "The \"rkyv\" feature must be enabled for this example to work. Try adding --features=\"rkyv\""
);

#[cfg(feature = "rkyv")]
mod example {
    use rkyv::Deserialize;

    #[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    struct Vertex {
        normal: normal_pack::EncodedUnitVector3F16,
    }

    pub fn run() {
        let normal = [-0.5082557, 0.54751796, 0.6647558];

        let vertex = Vertex {
            normal: normal_pack::EncodedUnitVector3F16::encode(normal),
        };

        let bytes = rkyv::to_bytes::<_, 4>(&vertex).unwrap();
        let archived = unsafe { rkyv::archived_root::<Vertex>(&bytes[..]) };
        let recasted: Vertex = archived.deserialize(&mut rkyv::Infallible).unwrap();
        let decoded_normal = recasted.normal.decode();

        println!("Bytes: {bytes:?}");
        println!("Recasted: {recasted:?}");
        println!("Decoded normal: {decoded_normal:#?}");
    }
}

fn main() {
    #[cfg(feature = "rkyv")]
    example::run();
}
