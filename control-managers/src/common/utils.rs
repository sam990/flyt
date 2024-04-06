use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::time::Duration;
use ipc_rs::MessageQueue;
use toml::Table;

use crate::common::api_commands::FlytApiCommand;
use crate::common::types::MqueueClientControlCommand;

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

    pub fn is_ping_active(mqueue: &MessageQueue, send_id: i64, recv_id: i64) -> bool {
        let cmd = MqueueClientControlCommand::new(FlytApiCommand::PING, "").as_bytes();
        mqueue.send(&cmd, send_id).unwrap();
        let response = mqueue.recv_type_timed(recv_id, Duration::from_secs(2));
        response.is_ok()
    }


    pub fn convert_bytes_to_u32(bytes: &[u8]) -> Option<u32> {
       if bytes.len() != 4 {
           return None;
       }
       Some(u32::from_be_bytes(bytes.try_into().unwrap()))    
    }

    pub fn load_config_file(config_path: &str) -> Table {
        let mut file = File::open(config_path).unwrap();
        let mut contents = String::new();
    
        file.read_to_string(&mut contents).unwrap();
    
        contents.parse::<Table>().unwrap()
    }
}