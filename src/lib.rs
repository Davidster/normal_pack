// TODO: add docs
// TODO: add license
// TODO: cleanup shaders

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(feature = "zerocopy", derive(zerocopy::AsBytes, zerocopy::FromBytes))]
pub struct EncodedUnitVector3([f32; 2]);

impl EncodedUnitVector3 {
    pub fn new(unit_vector: [f32; 3]) -> Self {
        let mut n = unit_vector;

        debug_assert!(
            (length_2(n) - 1.0).abs() < 0.0001,
            "Argument must be normalized"
        );

        let inv_sum = 1.0 / (n[0].abs() + n[1].abs() + n[2].abs());
        n[0] *= inv_sum;
        n[1] *= inv_sum;
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

    use crate::EncodedUnitVector3;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq)]
    #[cfg_attr(
        feature = "rkyv",
        derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
    )]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
    #[cfg_attr(feature = "zerocopy", derive(zerocopy::AsBytes, zerocopy::FromBytes))]
    pub struct EncodedUnitVector3F16([f16; 2]);

    impl EncodedUnitVector3F16 {
        pub fn new(unit_vector: [f32; 3]) -> Self {
            let encoded_f32 = EncodedUnitVector3::new(unit_vector);
            Self([
                f16::from_f32(encoded_f32.0[0]),
                f16::from_f32(encoded_f32.0[1]),
            ])
        }

        pub fn from_raw(raw: [f16; 2]) -> Self {
            Self(raw)
        }

        pub fn to_array(&self) -> [f32; 3] {
            EncodedUnitVector3::from_raw([self.0[0].to_f32(), self.0[1].to_f32()]).to_array()
        }

        pub fn raw(&self) -> [f16; 2] {
            self.0
        }
    }
}

#[cfg(feature = "half")]
pub use float16::EncodedUnitVector3F16;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(feature = "zerocopy", derive(zerocopy::AsBytes, zerocopy::FromBytes))]
pub struct EncodedUnitVector3U8([u8; 2]);

impl EncodedUnitVector3U8 {
    pub fn new(unit_vector: [f32; 3]) -> Self {
        let encoded_f32 = EncodedUnitVector3::new(unit_vector);
        Self([Self::to_u8(encoded_f32.0[0]), Self::to_u8(encoded_f32.0[1])])
    }

    pub fn from_raw(raw: [u8; 2]) -> Self {
        Self(raw)
    }

    pub fn to_array(&self) -> [f32; 3] {
        EncodedUnitVector3([Self::to_f32(self.0[0]), Self::to_f32(self.0[1])]).to_array()
    }

    pub fn raw(&self) -> [u8; 2] {
        self.0
    }

    #[inline]
    fn to_u8(from: f32) -> u8 {
        (((from + 1.0) * 0.5) * 255.0) as u8
    }

    #[inline]
    fn to_f32(from: u8) -> f32 {
        (from as f32 / 255.0) * 2.0 - 1.0
    }
}

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
        let expected_avg_error = 1.0932611e-5;
        let expected_max_error = 0.00048828125;
        test_error_rate_impl(
            |unit_vector| crate::EncodedUnitVector3::new(unit_vector).to_array(),
            expected_avg_error,
            expected_max_error,
        );
    }

    #[test]
    #[cfg(feature = "half")]
    fn test_error_rate_f16() {
        let expected_avg_error = 0.00013977697;
        let expected_max_error = 0.001035801;
        test_error_rate_impl(
            |unit_vector| crate::EncodedUnitVector3F16::new(unit_vector).to_array(),
            expected_avg_error,
            expected_max_error,
        );
    }

    #[test]
    fn test_error_rate_u8() {
        let expected_avg_error = 0.01223357;
        let expected_max_error = 0.03299934;
        test_error_rate_impl(
            |unit_vector| crate::EncodedUnitVector3U8::new(unit_vector).to_array(),
            expected_avg_error,
            expected_max_error,
        );
    }

    /// Loop through 100k unit vectors that are randomly distributed around the unit sphere
    /// and calculate the (max and average) error via the angle between the initial and decoded vector
    fn test_error_rate_impl<F>(codec: F, expected_avg_error: f32, expected_max_error: f32)
    where
        F: Fn([f32; 3]) -> [f32; 3],
    {
        let sample_size = 100_000;
        let unit_vectors = generate_unit_vectors(sample_size);

        let mut acc_error: f32 = 0.0;
        let mut max_error: f32 = 0.0;

        for unit_vector in unit_vectors {
            let decoded = codec(unit_vector);
            let error = angle_between(unit_vector, decoded);
            acc_error += error;
            max_error = max_error.max(error);
        }

        let avg_error = acc_error / sample_size as f32;

        assert_eq!(max_error, expected_max_error);
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

    fn angle_between(v1: [f32; 3], v2: [f32; 3]) -> f32 {
        ((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
            / (length_2(v1).sqrt() * length_2(v2).sqrt()))
        .clamp(-1.0, 1.0)
        .acos()
    }
}
