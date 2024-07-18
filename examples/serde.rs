use static_assertions::assert_cfg;

assert_cfg!(feature = "serde", "The \"serde\" feature must be enabled for this example to work. Try adding --features=\"serde\"");

#[cfg(feature = "serde")]
mod example {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct Vertex {
        normal: normal_pack::EncodedUnitVector3F16,
    }

    pub fn run() {
        let normal = [-0.5082557, 0.54751796, 0.6647558];

        let vertex = Vertex {
            normal: normal_pack::EncodedUnitVector3F16::encode(normal),
        };

        let json_string = serde_json::to_string_pretty(&vertex).unwrap();
        let parsed: Vertex = serde_json::from_str(&json_string).unwrap();
        let decoded_normal = parsed.normal.decode();

        println!("JSON: {json_string}");
        println!("Parsed: {parsed:#?}");
        println!("Parsed normal: {decoded_normal:#?}");
    }
}

fn main() {
    #[cfg(feature = "serde")]
    example::run();
}
