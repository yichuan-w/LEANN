/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! File operations

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{mem, io};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, BufReader, Write, Seek, SeekFrom};
use std::path::Path;

use crate::model::data_store::DatasetDto;

/// Read metadata of data file.
pub fn load_metadata_from_file(file_name: &str) -> std::io::Result<(usize, usize)> {
    let file = File::open(file_name)?;
    let mut reader = BufReader::new(file);

    let npoints = reader.read_i32::<LittleEndian>()? as usize;
    let ndims = reader.read_i32::<LittleEndian>()? as usize;

    Ok((npoints, ndims))
}

/// Read the deleted vertex ids from file.
pub fn load_ids_to_delete_from_file(file_name: &str) -> std::io::Result<(usize, Vec<u32>)> {
    // The first 4 bytes are the number of vector ids. 
    // The rest of the file are the vector ids in the format of usize. 
    // The vector ids are sorted in ascending order.
    let mut file = File::open(file_name)?;
    let num_ids = file.read_u32::<LittleEndian>()? as usize;
    
    let mut ids = Vec::with_capacity(num_ids);
    for _ in 0..num_ids {
        let id = file.read_u32::<LittleEndian>()?;
        ids.push(id);
    }
       
    Ok((num_ids, ids))
}

/// Copy data from file
/// # Arguments
/// * `bin_file` - filename where the data is
/// * `data` - destination dataset dto to which the data is copied
/// * `pts_offset` - offset of points. data will be loaded after this point in dataset
/// * `npts` - number of points read from bin_file
/// * `dim` - point dimension read from bin_file
/// * `rounded_dim` - rounded dimension (padding zero if it's > dim)
/// # Return 
/// * `npts` - number of points read from bin_file
/// * `dim` - point dimension read from bin_file
pub fn copy_aligned_data_from_file<T: Default + Copy>(
    bin_file: &str,
    dataset_dto: DatasetDto<T>,
    pts_offset: usize,
) -> std::io::Result<(usize, usize)> {
    let mut reader = File::open(bin_file)?;

    let npts = reader.read_i32::<LittleEndian>()? as usize;
    let dim = reader.read_i32::<LittleEndian>()? as usize;
    let rounded_dim = dataset_dto.rounded_dim;
    let offset = pts_offset * rounded_dim;

    for i in 0..npts {
        let data_slice = &mut dataset_dto.data[offset + i * rounded_dim..offset + i * rounded_dim + dim];
        let mut buf = vec![0u8; dim * mem::size_of::<T>()];
        reader.read_exact(&mut buf)?;
        
        let ptr = buf.as_ptr() as *const T;
        let temp_slice = unsafe { std::slice::from_raw_parts(ptr, dim) };
        data_slice.copy_from_slice(temp_slice);
        
        (i * rounded_dim + dim..i * rounded_dim + rounded_dim).for_each(|j| {
            dataset_dto.data[j] = T::default();
        });
    }

    Ok((npts, dim))
}

/// Open a file to write
/// # Arguments
/// * `writer` - mutable File reference
/// * `file_name` - file name
#[inline]
pub fn open_file_to_write(file_name: &str) -> std::io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .open(Path::new(file_name))
}

/// Delete a file
/// # Arguments
/// * `file_name` - file name
pub fn delete_file(file_name: &str) -> std::io::Result<()> {
    if file_exists(file_name) {
        fs::remove_file(file_name)?;
    }

    Ok(())
}

/// Check whether file exists or not
pub fn file_exists(filename: &str) -> bool {
    std::path::Path::new(filename).exists()
}

/// Save data to file
/// # Arguments
/// * `filename` - filename where the data is
/// * `data` - information data
/// * `npts` - number of points
/// * `ndims` - point dimension
/// * `aligned_dim` - aligned dimension
/// * `offset` - data offset in file
pub fn save_data_in_base_dimensions<T: Default + Copy>(
    filename: &str, 
    data: &mut [T], 
    npts: usize, 
    ndims: usize,
    aligned_dim: usize, 
    offset: usize,
) -> std::io::Result<usize> {
    let mut writer = open_file_to_write(filename)?;
    let npts_i32 = npts as i32;
    let ndims_i32 = ndims as i32;
    let bytes_written = 2 * std::mem::size_of::<u32>() + npts * ndims * (std::mem::size_of::<T>());

    writer.seek(std::io::SeekFrom::Start(offset as u64))?;
    writer.write_all(&npts_i32.to_le_bytes())?;
    writer.write_all(&ndims_i32.to_le_bytes())?;
    let data_ptr = data.as_ptr() as *const u8;
    for i in 0..npts {
        let middle_offset = i * aligned_dim * std::mem::size_of::<T>();
        let middle_slice = unsafe { std::slice::from_raw_parts(data_ptr.add(middle_offset), ndims * std::mem::size_of::<T>()) };
        writer.write_all(middle_slice)?;
    }
    writer.flush()?;
    Ok(bytes_written)
}

/// Read data file
/// # Arguments
/// * `bin_file` - filename where the data is
/// * `file_offset` - data offset in file
/// * `data` - information data
/// * `npts` - number of points
/// * `ndims` - point dimension
pub fn load_bin<T: Copy>(
    bin_file: &str, 
    file_offset: usize) -> std::io::Result<(Vec<T>, usize, usize)>
{    
    let mut reader = File::open(bin_file)?;
    reader.seek(std::io::SeekFrom::Start(file_offset as u64))?;
    let npts = reader.read_i32::<LittleEndian>()? as usize;
    let dim = reader.read_i32::<LittleEndian>()? as usize;

    let size = npts * dim * std::mem::size_of::<T>();
    let mut buf = vec![0u8; size];
    reader.read_exact(&mut buf)?;

    let ptr = buf.as_ptr() as *const T;
    let data = unsafe { std::slice::from_raw_parts(ptr, npts * dim)};

    Ok((data.to_vec(), npts, dim))
}

/// Get file size
pub fn get_file_size(filename: &str) -> io::Result<u64> {
    let reader = File::open(filename)?;
    let metadata = reader.metadata()?;
    Ok(metadata.len())
}

macro_rules! save_bin {
    ($name:ident, $t:ty, $write_func:ident) => {
        /// Write data into file
        pub fn $name(filename: &str, data: &[$t], num_pts: usize, dims: usize, offset: usize) -> std::io::Result<usize> {
            let mut writer = open_file_to_write(filename)?;

            println!("Writing bin: {}", filename);
            writer.seek(SeekFrom::Start(offset as u64))?;
            let num_pts_i32 = num_pts as i32;
            let dims_i32 = dims as i32;
            let bytes_written = num_pts * dims * mem::size_of::<$t>() + 2 * mem::size_of::<u32>();

            writer.write_i32::<LittleEndian>(num_pts_i32)?;
            writer.write_i32::<LittleEndian>(dims_i32)?;
            println!("bin: #pts = {}, #dims = {}, size = {}B", num_pts, dims, bytes_written);

            for item in data.iter() {
                writer.$write_func::<LittleEndian>(*item)?;
            }

            writer.flush()?;

            println!("Finished writing bin.");
            Ok(bytes_written)
        }
    };
}

save_bin!(save_bin_f32, f32, write_f32);
save_bin!(save_bin_u64, u64, write_u64);
save_bin!(save_bin_u32, u32, write_u32);

#[cfg(test)]
mod file_util_test {
    use crate::model::data_store::InmemDataset;
    use std::fs;
    use super::*;

    pub const DIM_8: usize = 8;

    #[test]
    fn load_metadata_test() {
        let file_name = "test_load_metadata_test.bin";
        let data = [200, 0, 0, 0, 128, 0, 0, 0]; // 200 and 128 in little endian bytes
        std::fs::write(file_name, data).expect("Failed to write sample file");
        match load_metadata_from_file(file_name) {
            Ok((npoints, ndims)) => {
                assert!(npoints == 200);
                assert!(ndims == 128);
            },
            Err(_e) => {},
        }
        fs::remove_file(file_name).expect("Failed to delete file");
    }

    #[test]
    fn load_data_test() {
        let file_name = "test_load_data_test.bin";
        //npoints=2, dim=8, 2 vectors [1.0;8] [2.0;8]
        let data: [u8; 72] = [2, 0, 0, 0, 8, 0, 0, 0, 
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 
            0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 
            0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 
            0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41]; 
        std::fs::write(file_name, data).expect("Failed to write sample file");

        let mut dataset = InmemDataset::<f32, DIM_8>::new(2, 1f32).unwrap();

        match copy_aligned_data_from_file(file_name, dataset.into_dto(), 0) {
            Ok((num_points, dim)) => {
                fs::remove_file(file_name).expect("Failed to delete file");
                assert!(num_points == 2);
                assert!(dim == 8);
                assert!(dataset.data.len() == 16);

                let first_vertex = dataset.get_vertex(0).unwrap();
                let second_vertex = dataset.get_vertex(1).unwrap();

                assert!(*first_vertex.vector() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
                assert!(*second_vertex.vector() == [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
            },
            Err(e) => {
                fs::remove_file(file_name).expect("Failed to delete file");
                panic!("{}", e)
            },
        }
    }

    #[test]
    fn open_file_to_write_test() {
        let file_name = "test_open_file_to_write_test.bin";
        let mut writer = File::create(file_name).unwrap();
        let data = [200, 0, 0, 0, 128, 0, 0, 0];
        writer.write(&data).expect("Failed to write sample file");

        let _ = open_file_to_write(file_name);

        fs::remove_file(file_name).expect("Failed to delete file");
    }

    #[test]
    fn delete_file_test() {
        let file_name = "test_delete_file_test.bin";
        let mut file = File::create(file_name).unwrap();
        writeln!(file, "test delete file").unwrap();

        let result = delete_file(file_name);

        assert!(result.is_ok());
        assert!(fs::metadata(file_name).is_err());
    }

    #[test]
    fn save_data_in_base_dimensions_test() {
        //npoints=2, dim=8
        let mut data: [u8; 72] = [2, 0, 0, 0, 8, 0, 0, 0, 
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 
            0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 
            0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 
            0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41]; 
        let num_points = 2;
        let dim = DIM_8;
        let data_file = "save_data_in_base_dimensions_test.data";
        match save_data_in_base_dimensions(data_file, &mut data, num_points, dim, DIM_8, 0) {
            Ok(num) => {
                assert!(file_exists(data_file));
                assert_eq!(num, 2 * std::mem::size_of::<u32>() + num_points * dim * std::mem::size_of::<u8>());
                fs::remove_file(data_file).expect("Failed to delete file");
            },
            Err(e) => {
                fs::remove_file(data_file).expect("Failed to delete file");
                panic!("{}", e)
            }
        }
    }

    #[test]
    fn save_bin_test() {
        let filename = "save_bin_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        let bytes_written = save_bin_u64(filename, &data, num_pts, dims, 0).unwrap();
        assert_eq!(bytes_written, 32);

        let mut file = File::open(filename).unwrap();
        let mut buffer = vec![];

        let npts_read = file.read_i32::<LittleEndian>().unwrap() as usize;
        let dims_read = file.read_i32::<LittleEndian>().unwrap() as usize;

        file.read_to_end(&mut buffer).unwrap();
        let data_read: Vec<u64> = buffer
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();

        std::fs::remove_file(filename).unwrap();

        assert_eq!(num_pts, npts_read);
        assert_eq!(dims, dims_read);
        assert_eq!(data, data_read);
    }

    #[test]
    fn load_bin_test() {
        let file_name = "load_bin_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        let bytes_written = save_bin_u64(file_name, &data, num_pts, dims, 0).unwrap();
        assert_eq!(bytes_written, 32);

        let (load_data, load_num_pts, load_dims) = load_bin::<u64>(file_name, 0).unwrap();
        assert_eq!(load_num_pts, num_pts);
        assert_eq!(load_dims, dims);
        assert_eq!(load_data, data);
        std::fs::remove_file(file_name).unwrap();
    }

    #[test]
    fn load_bin_offset_test() {
        let offset:usize = 32;
        let file_name = "load_bin_offset_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        let bytes_written = save_bin_u64(file_name, &data, num_pts, dims, offset).unwrap();
        assert_eq!(bytes_written, 32);

        let (load_data, load_num_pts, load_dims) = load_bin::<u64>(file_name, offset).unwrap();
        assert_eq!(load_num_pts, num_pts);
        assert_eq!(load_dims, dims);
        assert_eq!(load_data, data);
        std::fs::remove_file(file_name).unwrap();
    }
}

