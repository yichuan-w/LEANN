/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::{env, vec};

fn main() -> io::Result<()> {
    // Retrieve command-line arguments
    let args: Vec<String> = env::args().collect();

    // Check if the correct number of arguments is provided
    if args.len() != 4 {
        print_usage();
        return Ok(());
    }

    // Retrieve the input and output file paths from the arguments
    let input_file_path = &args[1];
    let item_count: usize = args[2].parse::<usize>().unwrap();
    let return_dimension: usize = args[3].parse::<usize>().unwrap();

    // Open the input file for reading
    let mut input_file = BufReader::new(File::open(input_file_path)?);

    // Read the first 8 bytes as metadata
    let mut metadata = [0; 8];
    input_file.read_exact(&mut metadata)?;

    // Extract the number of points and dimension from the metadata
    let _ = i32::from_le_bytes(metadata[..4].try_into().unwrap());
    let mut dimension: usize = (i32::from_le_bytes(metadata[4..].try_into().unwrap())) as usize;
    if return_dimension < dimension {
        dimension = return_dimension;
    }

    let mut float_array = Vec::<Vec<f32>>::with_capacity(item_count);

    // Process each data point
    for _ in 0..item_count {
        // Read one data point from the input file
        let mut buffer = vec![0; dimension * std::mem::size_of::<f32>()];
        match input_file.read_exact(&mut buffer) {
            Ok(()) => {
                let mut float_data = buffer
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<f32>>();

                let mut i = return_dimension;
                while i > dimension {
                    float_data.push(0.0);
                    i -= 1;
                }

                float_array.push(float_data);
            }
            Err(err) => {
                println!("Error: {}", err);
                break;
            }
        }
    }

    use base64::{engine::general_purpose, Engine as _};

    let encoded: Vec<u8> = bincode::serialize(&float_array).unwrap();
    let b64 = general_purpose::STANDARD.encode(encoded);
    println!("Float {}", b64);

    Ok(())
}

/// Prints the usage information
fn print_usage() {
    println!("Usage: program_name input_file <itemcount> <dimensions>");
    println!(
        "Itemcount is the number of items to convert. Expand to dimension if provided is smaller"
    );
}

