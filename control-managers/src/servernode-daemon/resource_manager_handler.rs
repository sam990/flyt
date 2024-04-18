use std::{io::{BufRead, BufReader, Write}, net::TcpStream, sync::{Arc, RwLock}};
use crate::{common::api_commands::FlytApiCommand, gpu_manager::GPU, virt_server_manager::VirtServerManager};
use crate::gpu_manager;



pub struct ResourceManagerHandler {
    resource_manager_stream: RwLock<Option<TcpStream>>,
    node_gpus: RwLock<Option<Vec<GPU>>>,
    virt_server_manager: Arc<VirtServerManager>,
}

impl ResourceManagerHandler {

    pub fn new( virt_server_manager: Arc<VirtServerManager> ) -> ResourceManagerHandler {
        ResourceManagerHandler {
            resource_manager_stream: RwLock::new(None),
            node_gpus: RwLock::new(None),
            virt_server_manager
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
                    buf.clear();
                    reader.read_line(&mut buf).unwrap();
                    buf = buf.trim().to_string();
                    let mut parts = buf.split(",");

                    if parts.clone().count() != 3 {
                        writer.write_all("400\nInvalid number of arguments\n".as_bytes()).unwrap();
                        continue;
                    }

                    let gpu_id: u32 = parts.next().unwrap().parse().unwrap();
                    let num_cores: u32 = parts.next().unwrap().parse().unwrap();
                    let memory: u64 = parts.next().unwrap().parse().unwrap();

                    let total_gpu_sm_cores = self.node_gpus.read().unwrap().as_ref().unwrap().iter().find(|gpu| gpu.gpu_id == gpu_id).unwrap().total_cores;

                    let ret = self.virt_server_manager.create_virt_server(gpu_id, memory, num_cores, total_gpu_sm_cores);
                    
                    match ret {
                        Ok(rpc_id) => {
                            writer.write_all(format!("200\n{}\n", rpc_id).as_bytes()).unwrap();

                        }
                        Err(e) => {
                            writer.write_all(format!("500\n{}\n", e).as_bytes()).unwrap();
                        }
                    }
                
                }

                FlytApiCommand::RMGR_SNODE_DEALLOC_VIRT_SERVER => {
                    buf.clear();
                    reader.read_line(&mut buf).unwrap();
                    let rpc_id: u64 = buf.trim().parse().unwrap();
                    let ret = self.virt_server_manager.remove_virt_server(rpc_id);
                    match ret {
                        Ok(_) => {
                            writer.write_all("200\nDone\n".as_bytes()).unwrap();
                        }
                        Err(e) => {
                            writer.write_all(format!("500\n{}\n", e).as_bytes()).unwrap();
                        }
                    }
                }

                FlytApiCommand::RMGR_SNODE_CHANGE_RESOURCES => {
                    buf.clear();
                    reader.read_line(&mut buf).unwrap();
                    let mut parts = buf.trim().split(",");
                    if parts.clone().count() != 3 {
                        writer.write_all("400\nInvalid number of arguments\n".as_bytes()).unwrap();
                        continue;
                    }

                    let rpc_id: u64 = parts.next().unwrap().parse().unwrap();
                    let num_cores: u32 = parts.next().unwrap().parse().unwrap();
                    let memory: u64 = parts.next().unwrap().parse().unwrap();

                    let ret = self.virt_server_manager.change_resources(rpc_id, num_cores, memory);
                    match ret {
                        Ok(_) => {
                            writer.write_all("200\nDone\n".as_bytes()).unwrap();
                        }
                        Err(e) => {
                            writer.write_all(format!("500\n{}\n", e).as_bytes()).unwrap();
                        }
                    }
                }

                _ => {
                    println!("Unknown command: {}", buf);
                }
            }
        }
    }


}