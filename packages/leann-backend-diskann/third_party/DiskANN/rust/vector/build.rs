/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
fn main() {
    println!("cargo:rerun-if-changed=distance.c");
    if cfg!(target_os = "macos") {
        std::env::set_var("CFLAGS", "-mavx2 -mfma -Wno-error -MP -O2 -D NDEBUG -D MKL_ILP64 -D USE_AVX2 -D USE_ACCELERATED_PQ -D NOMINMAX -D _TARGET_ARM_APPLE_DARWIN");

        cc::Build::new()
            .file("distance.c")
            .warnings_into_errors(true)
            .debug(false)
            .target("x86_64-apple-darwin")
            .compile("nativefunctions.lib");
    } else {
        std::env::set_var("CFLAGS", "/permissive- /MP /ifcOutput /GS- /W3 /Gy /Zi /Gm- /O2 /Ob2 /Zc:inline /fp:fast /D NDEBUG /D MKL_ILP64 /D USE_AVX2 /D USE_ACCELERATED_PQ /D NOMINMAX /fp:except- /errorReport:prompt /WX /openmp:experimental /Zc:forScope /GR /arch:AVX2 /Gd /Oy /Oi /MD /std:c++14 /FC /EHsc /nologo /Ot");
        // std::env::set_var("CFLAGS", "/permissive- /MP /ifcOutput /GS- /W3 /Gy /Zi /Gm- /Obd /Zc:inline /fp:fast /D DEBUG /D MKL_ILP64 /D USE_AVX2 /D USE_ACCELERATED_PQ /D NOMINMAX /fp:except- /errorReport:prompt /WX /openmp:experimental /Zc:forScope /GR /arch:AVX512 /Gd /Oy /Oi /MD /std:c++14 /FC /EHsc /nologo /Ot");

        cc::Build::new()
            .file("distance.c")
            .warnings_into_errors(true)
            .debug(false)
            .compile("nativefunctions");

        println!("cargo:rustc-link-arg=nativefunctions.lib");
    }
}

