use std::io::{BufRead, BufReader, Read, Write};
use ipc_rs::MessageQueue;

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

    pub fn send_ping_command(mqueue: &MessageQueue, send_id: i64) -> Result<(), ipc_rs::IpcError> {
        let bytes = MqueueClientControlCommand::new(FlytApiCommand::PING, "").as_bytes();
        mqueue.send(&bytes, send_id)
    }

    pub fn received_ping_response(mqueue: &MessageQueue, recv_id: i64) -> bool {
        mqueue.recv_type_nonblocking(recv_id).is_ok()
    }


    pub fn convert_bytes_to_u32(bytes: &[u8]) -> Option<u32> {
       if bytes.len() != 4 {
           return None;
       }
       Some(u32::from_be_bytes(bytes.try_into().unwrap()))    
    }
}