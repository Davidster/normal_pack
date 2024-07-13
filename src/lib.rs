use half::f16;

// TODO: add feature for serde, bytemuck and zerocopy support
// TODO: expose another version using f32 instead of f16?
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EncodedUnitVector([f16; 2]);

impl EncodedUnitVector {
    pub fn from_array(unit_vector: [f32; 3]) -> Self {
        // TODO: add debug assertion that it is normalized
        // TODO: accelerate with SIMD?
        let mut n = unit_vector;
        let sum = n[0].abs() + n[1].abs() + n[2].abs();
        n[0] /= sum;
        n[1] /= sum;
        n[2] /= sum;
        let mut result_vec2 = if n[2] >= 0.0 {
            [n[0], n[1]]
        } else {
            oct_wrap([n[0], n[1]])
        };
        result_vec2[0] = result_vec2[0] * 0.5 + 0.5;
        result_vec2[1] = result_vec2[1] * 0.5 + 0.5;
        Self([f16::from_f32(result_vec2[0]), f16::from_f32(result_vec2[1])])
    }

    pub fn raw(&self) -> [f16; 2] {
        self.0
    }
}

fn oct_wrap(v: [f32; 2]) -> [f32; 2] {
    [
        if v[0] >= 0.0 { 1.0 } else { -1.0 } * (1.0 - v[1].abs()),
        if v[1] >= 0.0 { 1.0 } else { -1.0 } * (1.0 - v[0].abs()),
    ]
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
