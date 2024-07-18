//! Compresses normal vectors (or any 3D unit vector) using [Octahedron encoding](https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/).
//!
//! This lossy compression scheme is able to achieve a compression ratio as high as 6:1 with an average error rate of less than 1 degree,
//! depending on which representation is chosen.
//!
//! #### Example:
//!
//! ```
//! let normal = [-0.5082557, 0.54751796, 0.6647558];
//!
//! let encoded = normal_pack::EncodedUnitVector3U8::encode(normal);
//! let decoded = encoded.decode();
//!
//! assert_eq!(decoded, [-0.52032965, 0.5473598, 0.6554802]);
//! ```
//!
//! #### Why compress my normals?
//!
//! It is common for 3D renderers to be bottlenecked by memory bandwidth, such as when loading normals from VRAM for high-poly meshes to supply to your vertex shader.
//! A smaller memory footprint for your normals corresponds to memory bandwidth savings and higher FPS in such scenarios.
//!
//! #### How bad is 1 degree of error?
//!
//! The `teapot` example generates a reference visual and contains the wgsl code required to decode the vector in a shader.
//!
//! ##### Standard [f32; 3] representation
//! ![teapot_packed_u8](https://github.com/user-attachments/assets/b16818d0-8020-477a-b6ec-99966eb1ae85)
//!
//! ##### Packed into a [u8; 2]
//! ![teapot_no_packing](https://github.com/user-attachments/assets/6e6ab8ad-37da-4be0-b8ef-e17c0ae9614f)
//!
//! *The skybox used in the example is the work of Emil Persson, aka Humus. [http://www.humus.name](http://www.humus.name)*
//!

/// A unit vector packed into an [f32; 2]
///
/// See the [module-level documentation](./index.html) for more details.
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
    /// Encodes the unit vector, stores the result in a new instance of this struct and returns it
    pub fn encode(unit_vector: [f32; 3]) -> Self {
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

    /// Decodes the unit vector and returns the result
    pub fn decode(&self) -> [f32; 3] {
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

    /// Stores the raw, encoded value in a new instance of this struct and returns it
    pub fn from_raw(raw: [f32; 2]) -> Self {
        Self(raw)
    }

    /// Returns the raw, encoded value stored by this struct
    pub fn raw(&self) -> [f32; 2] {
        self.0
    }
}

#[cfg(feature = "half")]
mod float16 {
    use half::f16;

    use crate::EncodedUnitVector3;

    /// A unit vector packed into a [[`half::f16`][half::f16]; 2]. The `half` feature must be enabled to use it.
    ///
    /// See the [module-level documentation](./index.html) for more details.
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
        /// Encodes the unit vector, stores the result in a new instance of this struct and returns it
        pub fn encode(unit_vector: [f32; 3]) -> Self {
            let encoded_f32 = EncodedUnitVector3::encode(unit_vector);
            Self([
                f16::from_f32(encoded_f32.0[0]),
                f16::from_f32(encoded_f32.0[1]),
            ])
        }

        /// Decodes the unit vector and returns the result
        pub fn decode(&self) -> [f32; 3] {
            EncodedUnitVector3::from_raw([self.0[0].to_f32(), self.0[1].to_f32()]).decode()
        }

        /// Stores the raw, encoded value in a new instance of this struct and returns it
        pub fn from_raw(raw: [f16; 2]) -> Self {
            Self(raw)
        }

        /// Returns the raw, encoded value stored by this struct
        pub fn raw(&self) -> [f16; 2] {
            self.0
        }
    }
}

#[cfg(feature = "half")]
pub use float16::EncodedUnitVector3F16;

/// A unit vector packed into a [u8; 2]
///
/// See the [module-level documentation](./index.html) for more details.
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
    /// Encodes the unit vector, stores the result in a new instance of this struct and returns it
    pub fn encode(unit_vector: [f32; 3]) -> Self {
        let encoded_f32 = EncodedUnitVector3::encode(unit_vector);
        Self([Self::to_u8(encoded_f32.0[0]), Self::to_u8(encoded_f32.0[1])])
    }

    /// Decodes the unit vector and returns the result
    pub fn decode(&self) -> [f32; 3] {
        EncodedUnitVector3([Self::to_f32(self.0[0]), Self::to_f32(self.0[1])]).decode()
    }

    /// Stores the raw, encoded value in a new instance of this struct and returns it
    pub fn from_raw(raw: [u8; 2]) -> Self {
        Self(raw)
    }

    /// Returns the raw, encoded value stored by this struct
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
            |unit_vector| crate::EncodedUnitVector3::encode(unit_vector).decode(),
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
            |unit_vector| crate::EncodedUnitVector3F16::encode(unit_vector).decode(),
            expected_avg_error,
            expected_max_error,
        );
    }

    #[test]
    fn test_error_rate_u8() {
        let expected_avg_error = 0.01223357;
        let expected_max_error = 0.03299934;
        test_error_rate_impl(
            |unit_vector| crate::EncodedUnitVector3U8::encode(unit_vector).decode(),
            expected_avg_error,
            expected_max_error,
        );
    }

    /// Loop through 100k unit vectors that are randomly distributed around the unit sphere
    /// and calculate the error in radians via the angle between the initial and decoded vector
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
