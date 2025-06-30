/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::io::{Write, Seek, SeekFrom};
use std::fs::{OpenOptions, File};
use std::path::Path;

pub struct CachedWriter {
    /// File writer
    writer: File,

    /// # bytes to cache for one shot write
    cache_size: u64,

    /// Underlying buf for cache
    cache_buf: Vec<u8>,

    /// Offset into cache_buf for cur_pos
    cur_off: u64,

    /// File size
    fsize: u64,
}

impl CachedWriter {
    pub fn new(filename: &str, cache_size: u64) -> std::io::Result<Self> {
        let writer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(Path::new(filename))?;
        
        if cache_size == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Cache size must be greater than 0"));
        }
        
        println!("Opened: {}, cache_size: {}", filename, cache_size);
        Ok(Self {
            writer,
            cache_size,
            cache_buf: vec![0; cache_size as usize],
            cur_off: 0,
            fsize: 0,
        })
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        // dump any remaining data in memory
        if self.cur_off > 0 {
            self.flush_cache()?;
        }

        self.writer.flush()?;
        println!("Finished writing {}B", self.fsize);
        Ok(())
    }

    pub fn get_file_size(&self) -> u64 {
        self.fsize
    }

    /// Writes n_bytes from write_buf to the underlying cache
    pub fn write(&mut self, write_buf: &[u8]) -> std::io::Result<()> {
        let n_bytes = write_buf.len() as u64;
        if n_bytes <= (self.cache_size - self.cur_off) {
            // case 1: cache can take all data
            self.cache_buf[(self.cur_off as usize)..((self.cur_off + n_bytes) as usize)].copy_from_slice(&write_buf[..n_bytes as usize]);
            self.cur_off += n_bytes;
        } else {
            // case 2: cache cant take all data
            // go to disk and write existing cache data
            self.writer.write_all(&self.cache_buf[..self.cur_off as usize])?;
            self.fsize += self.cur_off;
            // write the new data to disk
            self.writer.write_all(write_buf)?;
            self.fsize += n_bytes;
            // clear cache data and reset cur_off
            self.cache_buf.fill(0);
            self.cur_off = 0;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> std::io::Result<()> {
        self.flush_cache()?;
        self.writer.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    fn flush_cache(&mut self) -> std::io::Result<()> {
        self.writer.write_all(&self.cache_buf[..self.cur_off as usize])?;
        self.fsize += self.cur_off;
        self.cache_buf.fill(0);
        self.cur_off = 0;
        Ok(())
    }
}

impl Drop for CachedWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod cached_writer_test {
    use std::fs;

    use super::*;

    #[test]
    fn cached_writer_works() {
        let file_name = "cached_writer_works_test.bin";
        //npoints=2, dim=8, 2 vectors [1.0;8] [2.0;8]
        let data: [u8; 72] = [2, 0, 1, 2, 8, 0, 1, 3, 
            0x00, 0x01, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 
            0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 
            0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 
            0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x11, 0x80, 0x41]; 

        let mut writer = CachedWriter::new(file_name, 8).unwrap();
        assert_eq!(writer.get_file_size(), 0);
        assert_eq!(writer.cache_size, 8);
        assert_eq!(writer.get_file_size(), 0);

        let cache_all_buf = &data[0..4];
        writer.write(cache_all_buf).unwrap();
        assert_eq!(&writer.cache_buf[..4], cache_all_buf);
        assert_eq!(&writer.cache_buf[4..], vec![0; 4]);
        assert_eq!(writer.cur_off, 4);
        assert_eq!(writer.get_file_size(), 0);

        let write_all_buf = &data[4..10];
        writer.write(write_all_buf).unwrap();
        assert_eq!(writer.cache_buf, vec![0; 8]);
        assert_eq!(writer.cur_off, 0);
        assert_eq!(writer.get_file_size(), 10);

        fs::remove_file(file_name).expect("Failed to delete file");
    }
}

