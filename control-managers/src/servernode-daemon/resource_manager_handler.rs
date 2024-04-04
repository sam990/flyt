use std::{io::{BufRead, BufReader, Write}, net::TcpStream, sync::RwLock};
use crate::{common::api_commands::FlytApiCommand, gpu_manager::GPU};
use crate::gpu_manager;



pub struct ResourceManagerHandler {
    resource_manager_stream: RwLock<Option<TcpStream>>,
    node_gpus: RwLock<Option<Vec<GPU>>>
}

impl ResourceManagerHandler {

    pub fn new() -> ResourceManagerHandler {
        ResourceManagerHandler {
            resource_manager_stream: RwLock::new(None),
            node_gpus: RwLock::new(None)
        }
    }

    pub fn connect(&self, address: &str, port: u16) -> Result<(),String> {
        let stream = TcpStream::connect(format!("{}:{}", address, port));
        match stream {
            Ok(stream) => {
                self.resource_manager_stream.write().unwrap().replace(stream);
                Ok(())
            }
            Err(error) => {
                println!("Error connecting to server: {}", error);
                Err(error.to_string())
            }
        }
    }

    pub fn incomming_message_handler(&self) {
        if self.resource_manager_stream.read().unwrap().is_none() {
            return;
        }

        let mut writer = self.resource_manager_stream.read().unwrap().as_ref().unwrap().try_clone().unwrap();
        let reader_stream = writer.try_clone().unwrap();
        let mut buf = String::new();
        let mut reader = BufReader::new(reader_stream);

        loop {
            buf.clear();
            let read_size = reader.read_line(&mut buf).unwrap();

            if read_size == 0 {
                println!("Connection closed");
                break;
            }

            match buf.trim() {
                FlytApiCommand::RMGR_SNODE_SEND_GPU_INFO => {
                    let gpus = gpu_manager::get_all_gpus();
                    match gpus {
                        Some(gpus) => {
                            self.node_gpus.write().unwrap().replace(gpus.clone());
                            writer.write_all(format!("200\n{}\n", gpus.len()).as_bytes()).unwrap();
                            for gpu in gpus {
                                let message = format!("{},{},{},{},{},{}\n", gpu.gpu_id, gpu.name, gpu.memory, gpu.sm_cores, gpu.total_cores, gpu.max_clock);
                                writer.write_all(message.as_bytes()).unwrap();
                            }
                        }
                        None => {
                            let message = format!("500\nUnable to get gpu information\n");
                            writer.write_all(message.as_bytes()).unwrap();
                        }
                    }
                    
                }

                FlytApiCommand::RMGR_SNODE_ALLOC_VIRT_SERVER => {
                    let mut gpu_id = 0;
                    let mut num_cores = 0;
                    let mut memory = 0;
                    let mut virt_server_id = 0;

                    reader.read_line(&mut buf).unwrap();
                    let mut parts = buf.split(",");

                    if parts.count() != 3 {
                        writer.write_all("400\nInvalid number of arguments\n".as_bytes()).unwrap();
                        continue;
                    }

                    // gpu_id = parts.next().unwrap().parse().unwrap();
                    // num_cores = parts.next().unwrap().parse().unwrap();
                    // memory = parts.next().unwrap().parse().unwrap();

                    
        

                
                }

                _ => {
                    println!("Unknown command: {}", buf);
                }
            }
        }
    }


}