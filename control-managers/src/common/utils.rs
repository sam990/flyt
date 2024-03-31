use std::io::{BufRead, BufReader, Read, Write};
use crate::common::api_commands::FlytApiCommand;

pub struct Utils;

impl Utils {

    pub fn read_response<T: Read>(stream: &mut T, num_lines: u16 ) -> Vec<String> {
    
        let mut reader = BufReader::new(stream);
        let mut buf = String::new();
        
        let mut response = Vec::new();
    
        for _ in 0..num_lines {
            reader.read_line(&mut buf).unwrap();
            response.push(buf.trim().to_string());
            buf.clear();
        }
    
        response
    }

    pub fn is_stream_alive<T: Read + Write>(stream: &mut T) -> bool {
        stream.write_all(format!("{}\n", FlytApiCommand::PING).as_bytes()).unwrap();
        let mut reader = BufReader::new(stream);
        let mut buf = String::new();
        let read_size = reader.read_line(&mut buf).unwrap();
        read_size > 0
    }
}