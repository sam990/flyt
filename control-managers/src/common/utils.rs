use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::time::Duration;
use ipc_rs::MessageQueue;
use toml::Table;

use crate::common::api_commands::FlytApiCommand;
use crate::common::types::MqueueClientControlCommand;

pub struct Utils;
pub struct StreamUtils;

impl Utils {


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

impl StreamUtils {

    pub fn read_response<T: BufRead>(reader: &mut T, num_lines: u16 ) -> std::io::Result<Vec<String>> {
    
        let mut buf = String::new();
        
        let mut response = Vec::new();
    
        for _ in 0..num_lines {
            match reader.read_line(&mut buf) {
                Ok(_) => {
                    response.push(buf.trim().to_string());
                    buf.clear();
                }
                Err(error) => {
                    return Err(error);
                }
            }
        }
    
        Ok(response)
    }

    pub fn read_line<T: BufRead>(reader: &mut T) -> std::io::Result<String> {
        let mut buf = String::new();
        match reader.read_line(&mut buf) {
            Ok(_) => Ok(buf.trim().to_string()),
            Err(error) => {
                Err(error)
            }
        }
    }

    pub fn is_stream_alive<R: BufRead, T: Write>(reader: &mut R, stream: &mut T) -> bool {
        match stream.write_all(format!("{}\n", FlytApiCommand::PING).as_bytes()) {
            Ok(_) => {}
            Err(e) => {
                log::error!("Error writing to stream: {}", e);
                return false;
            }
        }
        let mut buf = String::new();
        let read_size = match reader.read_line(&mut buf) {
            Ok(read_len) => { read_len }
            Err(e) => {
                log::error!("Error reading from stream: {}", e);
                return false;
            }
        };
        read_size > 0
    }
    
    pub fn write_all<T: Write>(stream: &mut T, data: String) -> std::io::Result<()> {
        match stream.write_all(data.as_bytes()) {
            Ok(_) => Ok(()),
            Err(error) => {
                log::error!("Error writing to stream: {}", error);
                return Err(error)
            }
        }
    }
}
