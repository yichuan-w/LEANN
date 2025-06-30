/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#[cfg(test)]
mod e2e_test {

    #[repr(C, align(32))]
    pub struct F32Slice104([f32; 104]);

    #[repr(C, align(32))]
    pub struct F16Slice104([Half; 104]);

    use approx::assert_abs_diff_eq;

    use crate::half::Half;
    use crate::l2_float_distance::{distance_l2_vector_f16, distance_l2_vector_f32};

    fn no_vector_compare_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            let a_f32 = a[i];
            let b_f32 = b[i];
            let diff = a_f32 - b_f32;
            sum += diff * diff;
        }
        sum
    }

    fn no_vector_compare(a: &[Half], b: &[Half]) -> f32 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            let a_f32 = a[i].to_f32();
            let b_f32 = b[i].to_f32();
            let diff = a_f32 - b_f32;
            sum += diff * diff;
        }
        sum
    }

    #[test]
    fn avx2_matches_novector() {
        for i in 1..3 {
            let (f1, f2) = get_test_data(0, i);

            let distance_f32x8 = distance_l2_vector_f32::<104>(&f1.0, &f2.0);
            let distance = no_vector_compare_f32(&f1.0, &f2.0);

            assert_abs_diff_eq!(distance, distance_f32x8, epsilon = 1e-6);
        }
    }

    #[test]
    fn avx2_matches_novector_random() {
        let (f1, f2) = get_test_data_random();

        let distance_f32x8 = distance_l2_vector_f32::<104>(&f1.0, &f2.0);
        let distance = no_vector_compare_f32(&f1.0, &f2.0);

        assert_abs_diff_eq!(distance, distance_f32x8, epsilon = 1e-4);
    }

    #[test]
    fn avx_f16_matches_novector() {
        for i in 1..3 {
            let (f1, f2) = get_test_data_f16(0, i);
            let _a_slice = f1.0.map(|x| x.to_f32().to_string()).join(", ");
            let _b_slice = f2.0.map(|x| x.to_f32().to_string()).join(", ");

            let expected = no_vector_compare(f1.0[0..].as_ref(), f2.0[0..].as_ref());
            let distance_f16x8 = distance_l2_vector_f16::<104>(&f1.0, &f2.0);

            assert_abs_diff_eq!(distance_f16x8, expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn avx_f16_matches_novector_random() {
        let (f1, f2) = get_test_data_f16_random();

        let expected = no_vector_compare(f1.0[0..].as_ref(), f2.0[0..].as_ref());
        let distance_f16x8 = distance_l2_vector_f16::<104>(&f1.0, &f2.0);

        assert_abs_diff_eq!(distance_f16x8, expected, epsilon = 1e-4);
    }

    fn get_test_data_f16(i1: usize, i2: usize) -> (F16Slice104, F16Slice104) {
        let (a_slice, b_slice) = get_test_data(i1, i2);
        let a_data = a_slice.0.iter().map(|x| Half::from_f32(*x));
        let b_data = b_slice.0.iter().map(|x| Half::from_f32(*x));

        (
            F16Slice104(a_data.collect::<Vec<Half>>().try_into().unwrap()),
            F16Slice104(b_data.collect::<Vec<Half>>().try_into().unwrap()),
        )
    }

    fn get_test_data(i1: usize, i2: usize) -> (F32Slice104, F32Slice104) {
        use base64::{engine::general_purpose, Engine as _};

        let b64 = general_purpose::STANDARD.decode(TEST_DATA).unwrap();

        let decoded: Vec<Vec<f32>> = bincode::deserialize(&b64).unwrap();
        debug_assert!(decoded.len() > i1);
        debug_assert!(decoded.len() > i2);

        let mut f1 = F32Slice104([0.0; 104]);
        let v1 = &decoded[i1];
        debug_assert!(v1.len() == 104);
        f1.0.copy_from_slice(v1);

        let mut f2 = F32Slice104([0.0; 104]);
        let v2 = &decoded[i2];
        debug_assert!(v2.len() == 104);
        f2.0.copy_from_slice(v2);

        (f1, f2)
    }

    fn get_test_data_f16_random() -> (F16Slice104, F16Slice104) {
        let (a_slice, b_slice) = get_test_data_random();
        let a_data = a_slice.0.iter().map(|x| Half::from_f32(*x));
        let b_data = b_slice.0.iter().map(|x| Half::from_f32(*x));

        (
            F16Slice104(a_data.collect::<Vec<Half>>().try_into().unwrap()),
            F16Slice104(b_data.collect::<Vec<Half>>().try_into().unwrap()),
        )
    }

    fn get_test_data_random() -> (F32Slice104, F32Slice104) {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut f1 = F32Slice104([0.0; 104]);

        for i in 0..104 {
            f1.0[i] = rng.gen_range(-1.0..1.0);
        }

        let mut f2 = F32Slice104([0.0; 104]);

        for i in 0..104 {
            f2.0[i] = rng.gen_range(-1.0..1.0);
        }

        (f1, f2)
    }

    const TEST_DATA: &str = "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";
}

