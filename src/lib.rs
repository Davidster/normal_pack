use half::f16;

// TODO: add feature for serde, bytemuck and zerocopy support
// TODO: expose another version using f32 instead of f16?
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EncodedUnitVector([f16; 2]);

impl EncodedUnitVector {
    pub fn from_array_f32(mut unit_vector: [f32; 3]) {
        // TODO: add debug assertion that it is normalized
        n = n / (n.x.abs() + n.y.abs() + n.z.abs());
        let mut result_vec2 = if n.z >= 0.0 { n.xy() } else { oct_wrap(n.xy()) };
        result_vec2 = result_vec2 * Vec2::splat(0.5) + Vec2::splat(0.5);
        [f16::from(result_vec2.x), f16::from(result_vec2.y)]
    }
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
