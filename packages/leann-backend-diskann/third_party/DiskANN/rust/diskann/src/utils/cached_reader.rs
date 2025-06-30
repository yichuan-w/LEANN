/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::fs::File;
use std::io::{Seek, Read};

use crate::common::{ANNResult, ANNError};

/// Sequential cached reads
pub struct CachedReader {
    /// File reader
    reader: File,

    /// # bytes to cache in one shot read
    cache_size: u64,

    /// Underlying buf for cache
    cache_buf: Vec<u8>,

    /// Offset into cache_buf for cur_pos
    cur_off: u64,

    /// File size
    fsize: u64,
}

impl CachedReader {
    pub fn new(filename: &str, cache_size: u64) -> std::io::Result<Self> {
        let mut reader = File::open(filename)?;
        let metadata = reader.metadata()?;
        let fsize = metadata.len();

        let cache_size = cache_size.min(fsize);
        let mut cache_buf = vec![0; cache_size as usize];
        reader.read_exact(&mut cache_buf)?;
        println!("Opened: {}, size: {}, cache_size: {}", filename, fsize, cache_size);

        Ok(Self {
            reader,
            cache_size,
            cache_buf,
            cur_off: 0,
            fsize,
        })
    }

    pub fn get_file_size(&self) -> u64 {
        self.fsize
    }

    pub fn read(&mut self, read_buf: &mut [u8]) -> ANNResult<()> {
        let n_bytes = read_buf.len() as u64;
        if n_bytes <= (self.cache_size - self.cur_off) {
            // case 1: cache contains all data
            read_buf.copy_from_slice(&self.cache_buf[(self.cur_off as usize)..(self.cur_off as usize + n_bytes as usize)]);
            self.cur_off += n_bytes;
        } else {
            // case 2: cache contains some data
            let cached_bytes = self.cache_size - self.cur_off;
            if n_bytes - cached_bytes > self.fsize - self.reader.stream_position()? {
                return Err(ANNError::log_index_error(format!(
                    "Reading beyond end of file, n_bytes: {} cached_bytes: {} fsize: {} current pos: {}", 
                    n_bytes, cached_bytes, self.fsize, self.reader.stream_position()?))
                );
            }

            read_buf[..cached_bytes as usize].copy_from_slice(&self.cache_buf[self.cur_off as usize..]);
            // go to disk and fetch more data
            self.reader.read_exact(&mut read_buf[cached_bytes as usize..])?;
            // reset cur off
            self.cur_off = self.cache_size;
            
            let size_left = self.fsize - self.reader.stream_position()?;
            if size_left >= self.cache_size {
                self.reader.read_exact(&mut self.cache_buf)?;
                self.cur_off = 0;
            }
            // note that if size_left < cache_size, then cur_off = cache_size,
            // so subsequent reads will all be directly from file
        }
        Ok(())
    }

    pub fn read_u32(&mut self) -> ANNResult<u32> {
        let mut bytes = [0u8; 4];
        self.read(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod cached_reader_test {
    use std::fs;

    use super::*;

    #[test]
    fn cached_reader_works() {
        let file_name = "cached_reader_works_test.bin";
        //npoints=2, dim=8, 2 vectors [1.0;8] [2.0;8]
        let data: [u8; 72] = [2, 0, 1, 2, 8, 0, 1, 3, 
            0x00, 0x01, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 
            0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 
            0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 
            0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x11, 0x80, 0x41]; 
        std::fs::write(file_name, data).expect("Failed to write sample file");

        let mut reader = CachedReader::new(file_name, 8).unwrap();
        assert_eq!(reader.get_file_size(), 72);
        assert_eq!(reader.cache_size, 8);

        let mut all_from_cache_buf = vec![0; 4];
        reader.read(all_from_cache_buf.as_mut_slice()).unwrap();
        assert_eq!(all_from_cache_buf, [2, 0, 1, 2]);
        assert_eq!(reader.cur_off, 4);

        let mut partial_from_cache_buf = vec![0; 6];
        reader.read(partial_from_cache_buf.as_mut_slice()).unwrap();
        assert_eq!(partial_from_cache_buf, [8, 0, 1, 3, 0x00, 0x01]);
        assert_eq!(reader.cur_off, 0);

        let mut over_cache_size_buf = vec![0; 60];
        reader.read(over_cache_size_buf.as_mut_slice()).unwrap();
        assert_eq!(
            over_cache_size_buf, 
            [0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 
            0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 
            0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 
            0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x11]
        );

        let mut remaining_less_than_cache_size_buf = vec![0; 2];
        reader.read(remaining_less_than_cache_size_buf.as_mut_slice()).unwrap();
        assert_eq!(remaining_less_than_cache_size_buf, [0x80, 0x41]);
        assert_eq!(reader.cur_off, reader.cache_size);

        fs::remove_file(file_name).expect("Failed to delete file");
    }

    #[test]
    #[should_panic(expected = "n_bytes: 73 cached_bytes: 8 fsize: 72 current pos: 8")]
    fn failed_for_reading_beyond_end_of_file() {
        let file_name = "failed_for_reading_beyond_end_of_file_test.bin";
        //npoints=2, dim=8, 2 vectors [1.0;8] [2.0;8]
        let data: [u8; 72] = [2, 0, 1, 2, 8, 0, 1, 3, 
            0x00, 0x01, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 
            0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 
            0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 
            0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x11, 0x80, 0x41]; 
        std::fs::write(file_name, data).expect("Failed to write sample file");

        let mut reader = CachedReader::new(file_name, 8).unwrap();
        fs::remove_file(file_name).expect("Failed to delete file");
        
        let mut over_size_buf = vec![0; 73];
        reader.read(over_size_buf.as_mut_slice()).unwrap();
    }
}

