/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use half::{bf16, f16};
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, BufReader, BufWriter};

enum F16OrBF16 {
    F16(f16),
    BF16(bf16),
}

fn main() -> io::Result<()> {
    // Retrieve command-line arguments
    let args: Vec<String> = env::args().collect();

    match args.len() {
        3|4|5|6=> {},
        _ => {
            print_usage();
            std::process::exit(1);
        }
    }

    // Retrieve the input and output file paths from the arguments
    let input_file_path = &args[1];
    let output_file_path = &args[2];
    let use_f16 = args.len() >= 4 && args[3] == "f16";
    let save_as_float = args.len() >= 5 && args[4] == "save_as_float";
    let batch_size = if args.len() >= 6 { args[5].parse::<i32>().unwrap() } else { 100000 };
    println!("use_f16: {}", use_f16);
    println!("save_as_float: {}", save_as_float);
    println!("batch_size: {}", batch_size);

    // Open the input file for reading
    let mut input_file = BufReader::new(File::open(input_file_path)?);

    // Open the output file for writing
    let mut output_file = BufWriter::new(OpenOptions::new().write(true).create(true).open(output_file_path)?);

    // Read the first 8 bytes as metadata
    let mut metadata = [0; 8];
    input_file.read_exact(&mut metadata)?;

    // Write the metadata to the output file
    output_file.write_all(&metadata)?;

    // Extract the number of points and dimension from the metadata
    let num_points = i32::from_le_bytes(metadata[..4].try_into().unwrap());
    let dimension = i32::from_le_bytes(metadata[4..].try_into().unwrap());
    let num_batches = num_points / batch_size;
    // Calculate the size of one data point in bytes
    let data_point_size = (dimension * 4 * batch_size) as usize;
    let mut batches_processed = 0;
    let numbers_to_print = 2;
    let mut numbers_printed = 0; 
    let mut num_fb16_wins = 0;
    let mut num_f16_wins = 0;
    let mut bf16_overflow = 0;
    let mut f16_overflow = 0;

    // Process each data point
    for _ in 0..num_batches {
        // Read one data point from the input file
        let mut buffer = vec![0; data_point_size];
        match input_file.read_exact(&mut buffer){
            Ok(()) => {
                // Convert the float32 data to bf16
                let half_data: Vec<F16OrBF16> = buffer
                .chunks_exact(4)
                .map(|chunk| {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let converted_bf16 = bf16::from_f32(value);
                    let converted_f16 = f16::from_f32(value);
                    let distance_f16 = (converted_f16.to_f32() - value).abs();
                    let distance_bf16 = (converted_bf16.to_f32() - value).abs();

                    if distance_f16 < distance_bf16 {
                        num_f16_wins += 1;
                    } else {
                        num_fb16_wins += 1;
                    }

                    if (converted_bf16 == bf16::INFINITY) || (converted_bf16 == bf16::NEG_INFINITY) {
                        bf16_overflow += 1;
                    }

                    if (converted_f16 == f16::INFINITY) || (converted_f16 == f16::NEG_INFINITY) {
                        f16_overflow += 1;
                    }

                    if numbers_printed < numbers_to_print {
                        numbers_printed += 1;
                        println!("f32 value: {} f16 value: {} | distance {},  bf16 value: {} | distance {},",
                        value, converted_f16, converted_f16.to_f32() - value, converted_bf16, converted_bf16.to_f32() - value);
                    }
                    
                    if use_f16 {
                        F16OrBF16::F16(converted_f16)
                    } else {
                        F16OrBF16::BF16(converted_bf16)
                    }
                })
                .collect();

            batches_processed += 1;

            match save_as_float {
                true => {
                    for float_val in half_data {
                        match float_val {
                            F16OrBF16::F16(f16_val) => output_file.write_all(&f16_val.to_f32().to_le_bytes())?,
                            F16OrBF16::BF16(bf16_val) => output_file.write_all(&bf16_val.to_f32().to_le_bytes())?,
                        }
                    }
                }
                false => {
                    for float_val in half_data {
                        match float_val {
                            F16OrBF16::F16(f16_val) => output_file.write_all(&f16_val.to_le_bytes())?,
                            F16OrBF16::BF16(bf16_val) => output_file.write_all(&bf16_val.to_le_bytes())?,
                        }
                    }
                }
             }

            // Print the number of points processed
            println!("Processed {} points out of {}", batches_processed * batch_size, num_points);
            }
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                println!("Conversion completed! {} of times f16 wins | overflow count {}, {} of times bf16 wins | overflow count{}",
                 num_f16_wins, f16_overflow, num_fb16_wins, bf16_overflow);
                break;
            }
            Err(err) => {
                println!("Error: {}", err);
                break;
            }
        };
    }

    Ok(())
}

/// Prints the usage information
fn print_usage() {
    println!("Usage: program_name input_file output_file [f16] [save_as_float] [batch_size]]");
    println!("specify f16 to downscale to f16. otherwise, downscale to bf16.");
    println!("specify save_as_float to downcast to f16 or bf16, and upcast to float before saving the output data. otherwise, the data will be saved as half type.");
    println!("specify the batch_size as a int, the default value is 100000.");
}

