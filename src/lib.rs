// TODO: add feature for serde, bytemuck and zerocopy support

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EncodedUnitVector([f32; 2]);

impl EncodedUnitVector {
    pub fn from_array(unit_vector: [f32; 3]) -> Self {
        debug_assert!(
            (length_2(unit_vector) - 1.0).abs() < 0.0001,
            "EncodedUnitVector is only compatible with unit vectors"
        );

        let mut n = unit_vector;
        let inv_sum = 1.0 / (n[0].abs() + n[1].abs() + n[2].abs());
        n[0] *= inv_sum;
        n[1] *= inv_sum;
        n[2] *= inv_sum;
        if n[2] < 0.0 {
            let x = n[0];
            n[0] = if n[0] >= 0.0 { 1.0 } else { -1.0 } * (1.0 - n[1].abs());
            n[1] = if n[1] >= 0.0 { 1.0 } else { -1.0 } * (1.0 - x.abs());
        }
        Self([n[0], n[1]])
    }

    pub fn from_raw(raw: [f32; 2]) -> Self {
        Self(raw)
    }

    pub fn to_array(&self) -> [f32; 3] {
        let x = self.0[0];
        let y = self.0[1];
        let z = 1.0 - x.abs() - y.abs();
        let t = (-z).max(0.0);
        normalize([
            x + if x >= 0.0 { -t } else { t },
            y + if y >= 0.0 { -t } else { t },
            z,
        ])
    }

    pub fn raw(&self) -> [f32; 2] {
        self.0
    }
}

#[cfg(feature = "half")]
mod float16 {
    use half::f16;

    use crate::EncodedUnitVector;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    pub struct EncodedUnitVectorF16([f16; 2]);

    impl EncodedUnitVectorF16 {
        pub fn from_array(unit_vector: [f32; 3]) -> Self {
            let encoded_f32 = EncodedUnitVector::from_array(unit_vector);
            Self([
                f16::from_f32(encoded_f32.0[0]),
                f16::from_f32(encoded_f32.0[1]),
            ])
        }

        pub fn from_raw(raw: [f16; 2]) -> Self {
            Self(raw)
        }

        pub fn to_array(&self) -> [f32; 3] {
            EncodedUnitVector::from_raw([self.0[0].to_f32(), self.0[1].to_f32()]).to_array()
        }

        pub fn raw(&self) -> [f16; 2] {
            self.0
        }
    }
}

#[cfg(feature = "half")]
pub use float16::EncodedUnitVectorF16;

#[inline]
fn length_2(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let inv_length = 1.0 / length_2(v).sqrt();
    [v[0] * inv_length, v[1] * inv_length, v[2] * inv_length]
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::{length_2, normalize};

    #[test]
    fn test_error_rate_f32() {
        let expected_avg_error = 9.62025e-7;
        test_error_rate_impl(
            |unit_vector| crate::EncodedUnitVector::from_array(unit_vector).to_array(),
            expected_avg_error,
        );
    }

    #[test]
    #[cfg(feature = "half")]
    fn test_error_rate_f16() {
        let expected_avg_error = 0.00248607;
        test_error_rate_impl(
            |unit_vector| crate::EncodedUnitVectorF16::from_array(unit_vector).to_array(),
            expected_avg_error,
        );
    }

    fn test_error_rate_impl<F>(codec: F, expected_avg_error: f32)
    where
        F: Fn([f32; 3]) -> [f32; 3],
    {
        let count = 100000;
        let unit_vectors = generate_unit_vectors(count);

        let mut error_x = 0.0;
        let mut error_y = 0.0;
        let mut error_z = 0.0;

        for unit_vector in unit_vectors {
            let decoded = codec(unit_vector);

            error_x += ((unit_vector[0] - decoded[0]) / unit_vector[0]).abs();
            error_y += ((unit_vector[1] - decoded[1]) / unit_vector[1]).abs();
            error_z += ((unit_vector[2] - decoded[2]) / unit_vector[2]).abs();
        }

        error_x /= count as f32;
        error_y /= count as f32;
        error_z /= count as f32;

        let avg_error = length_2([error_x, error_y, error_z]).sqrt();
        assert_eq!(avg_error, expected_avg_error);
    }

    fn generate_unit_vectors(count: usize) -> Vec<[f32; 3]> {
        // produce the same points by using the same seed every time with a deterministic and portable rng
        let mut rng = rand_chacha::ChaCha8Rng::from_seed([0; 32]);

        (0..count)
            .map(|_| {
                normalize([
                    rng.gen::<f32>() * 2.0 - 1.0,
                    rng.gen::<f32>() * 2.0 - 1.0,
                    rng.gen::<f32>() * 2.0 - 1.0,
                ])
            })
            .collect()
    }
}
