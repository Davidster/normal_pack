use static_assertions::assert_cfg;

assert_cfg!(feature = "serde", "The \"serde\" feature must be enabled for this example to work. Try adding --features=\"serde\"");

#[cfg(feature = "serde")]
mod example {
    use normal_pack::EncodedUnitVector3F16;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct Vertex {
        normal: EncodedUnitVector3F16,
    }

    pub fn run() {
        let normal = [-0.5082557, 0.54751796, 0.6647558];

        let vertex = Vertex {
            normal: EncodedUnitVector3F16::new(normal),
        };

        let json_string = serde_json::to_string_pretty(&vertex).unwrap();
        let parsed: Vertex = serde_json::from_str(&json_string).unwrap();
        let decoded = vertex.normal.to_array();

        println!("JSON: {json_string}");
        println!("Parsed: {parsed:#?}");
        println!("Decoded normal: {decoded:#?}");
    }
}

fn main() {
    #[cfg(feature = "serde")]
    example::run();
}
