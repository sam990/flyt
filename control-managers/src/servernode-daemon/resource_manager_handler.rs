use std::{io::{BufRead, BufReader, Write}, net::TcpStream, sync::{Arc, RwLock}};
use crate::{common::api_commands::FlytApiCommand, gpu_manager::GPUManager, virt_server_manager::VirtServerManager, common::utils::StreamUtils};

macro_rules! stream_clone {
    ($stream:expr) => {
        match $stream.try_clone() {
            Ok(stream) => stream,
            Err(e) => {
                log::error!("Error cloning stream: {}", e);
                return;
            }
        }
    };
}

macro_rules! stream_write {
    ($stream:expr, $message:expr) => {
        match $stream.write_all($message.as_bytes()) {
            Ok(_) => {},
            Err(e) => {
                log::error!("Error writing to stream: {}", e);
                return;
            }
        }
    };
}

macro_rules! stream_read_line {
    ($stream:expr) => {
        match StreamUtils::read_line(&mut $stream) {
            Ok(data) => data,
            Err(e) => {
                log::error!("Error reading from stream: {}", e);
                return;
            }
        }
    };
}

pub struct ResourceManagerHandler {
    resource_manager_stream: RwLock<Option<TcpStream>>,
    virt_server_manager: Arc<VirtServerManager>,
    gpu_manager: GPUManager
}

impl ResourceManagerHandler {

    pub fn new( virt_server_manager: Arc<VirtServerManager>, gpu_manager: GPUManager ) -> ResourceManagerHandler {
        ResourceManagerHandler {
            resource_manager_stream: RwLock::new(None),
            virt_server_manager,
            gpu_manager
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
                log::error!("Error connecting to server: {}", error);
                Err(error.to_string())
            }
        }
    }

    pub fn incomming_message_handler(&mut self) {
        if self.resource_manager_stream.read().unwrap().is_none() {
            return;
        }

        let mut writer = stream_clone!(self.resource_manager_stream.read().unwrap().as_ref().unwrap());
        let reader_stream = stream_clone!(writer);
        let mut buf = String::new();
        let mut reader = BufReader::new(reader_stream);

        loop {
            buf.clear();
            let read_size = match reader.read_line(&mut buf) {
                Ok(size) => size,
                Err(e) => {
                    log::error!("Error reading from stream: {}", e);
                    break;
                }
            };

            if read_size == 0 {
                log::error!("Connection closed");
                break;
            }

            match buf.trim() {
                FlytApiCommand::RMGR_SNODE_SEND_GPU_INFO => {
                    log::info!("Got send gpu info command");
                    let gpus = self.gpu_manager.get_all_gpus();
                    match gpus {
                        Some(gpus) => {
                            stream_write!(writer, format!("200\n{}\n", gpus.len()));
                            for gpu in gpus {
                                let message = format!("{},{},{},{},{},{}\n", gpu.gpu_id, gpu.name, gpu.memory, gpu.sm_cores, gpu.total_cores, gpu.max_clock);
                                stream_write!(writer, message);
                            }
                        }
                        None => {
                            let message = format!("500\nUnable to get gpu information\n");
                            stream_write!(writer, message);
                        }
                    }
                    
                }

                FlytApiCommand::RMGR_SNODE_ALLOC_VIRT_SERVER => {

                    log::info!("Got allocate virt server command");

                    let args = stream_read_line!(reader);
                    let parts = args.split(",").collect::<Vec<&str>>();

                    if parts.len() != 3 {
                        log::error!("Invalid number of arguments: {:?}", parts);
                        stream_write!(writer, "400\nInvalid number of arguments\n".to_string());
                        continue;
                    }

                    let gpu_id = parts[0].parse::<u32>();
                    let num_cores = parts[1].parse::<u32>();
                    let memory = parts[2].parse::<u64>();

                    if gpu_id.is_err() || num_cores.is_err() || memory.is_err() {
                        stream_write!(writer, "400\nInvalid arguments\n".to_string());
                        continue;
                    }

                    let gpu_id = gpu_id.unwrap();
                    let num_cores = num_cores.unwrap();
                    let memory = memory.unwrap();

                    log::info!("Allocating virt server: gpu_id: {}, num_cores: {}, memory: {}", gpu_id, num_cores, memory);

                    if self.gpu_manager.get_gpu(gpu_id).is_none() {
                        log::error!("GPU not found: {}", gpu_id);
                        stream_write!(writer, "400\nGPU not found\n".to_string());
                        continue;
                    }


                    let ret = self.virt_server_manager.create_virt_server(gpu_id, memory, num_cores);
                    
                    match ret {
                        Ok(rpc_id) => {
                            log::info!("Virt server created: {}", rpc_id);
                            stream_write!(writer, format!("200\n{}\n", rpc_id));

                        }
                        Err(e) => {
                            log::info!("Error creating virt server: {}", e);
                            stream_write!(writer, format!("500\n{}\n", e));
                        }
                    }
                
                }

                FlytApiCommand::RMGR_SNODE_CHECKPOINT => {
                    log::info!("Got checkpoint virt server command");

                    let rpc_id = stream_read_line!(reader).parse::<u64>();
                    if rpc_id.is_err() {
                        log::error!("Invalid rpc_id: {:?}", rpc_id);
                        stream_write!(writer, "400\nInvalid rpc_id\n".to_string());
                        continue;
                    }

                    let rpc_id = rpc_id.unwrap();
                    let ckp_path = stream_read_line!(reader);

                    log::info!("Checkpointing virt server: {} at path {}", rpc_id, ckp_path);
                    let ret = self.virt_server_manager.checkpoint_virt_server(rpc_id, ckp_path.as_str());
                    match ret {
                        Ok(_) => {
                            log::info!("Virt server checkpointed");
                            stream_write!(writer, format!("200\n{}\n", ckp_path));
                        }
                        Err(e) => {
                            log::error!("Error checkpointing virt server: {}", e);
                            stream_write!(writer, format!("500\n{}\n", e));
                        }
                    }
                }

                FlytApiCommand::RMGR_SNODE_RESTORE => {
                    log::info!("Got restore virt server command");

                    let rpc_id = stream_read_line!(reader).parse::<u64>();
                    if rpc_id.is_err() {
                        log::error!("Invalid rpc_id: {:?}", rpc_id);
                        stream_write!(writer, "400\nInvalid rpc_id\n".to_string());
                        continue;
                    }

                    let rpc_id = rpc_id.unwrap();
                    let ckp_path = stream_read_line!(reader);

                    log::info!("Restoring virt server: {} at path {}", rpc_id, ckp_path);
                    let ret = self.virt_server_manager.restore_virt_server(rpc_id, ckp_path.as_str());
                    match ret {
                        Ok(_) => {
                            log::info!("Virt server restored");
                            stream_write!(writer, format!("200\n{}\n", ckp_path));
                        }
                        Err(e) => {
                            log::error!("Error restoring virt server: {}", e);
                            stream_write!(writer, format!("500\n{}\n", e));
                        }
                    }
                }

                FlytApiCommand::RMGR_SNODE_DEALLOC_VIRT_SERVER => {
                    
                    log::info!("Got deallocate virt server command");

                    let rpc_id = stream_read_line!(reader).parse::<u64>();
                    if rpc_id.is_err() {
                        log::error!("Invalid rpc_id: {:?}", rpc_id);
                        stream_write!(writer, "400\nInvalid rpc_id\n".to_string());
                        continue;
                    }
                    let rpc_id = rpc_id.unwrap();
                    log::info!("Deallocating virt server: {}", rpc_id);
                    let ret = self.virt_server_manager.remove_virt_server(rpc_id);
                    match ret {
                        Ok(_) => {
                            log::info!("Virt server deallocated");
                            stream_write!(writer, "200\nDone\n".to_string());
                        }
                        Err(e) => {
                            log::error!("Error deallocating virt server: {}", e);
                            stream_write!(writer, format!("500\n{}\n", e));
                        }
                    }
                }

                FlytApiCommand::RMGR_SNODE_CHANGE_RESOURCES => {
                    log::info!("Got change resources command");

                    let args = stream_read_line!(reader);
                    let parts = args.split(",").collect::<Vec<&str>>();

                    if parts.len() != 3 {
                        log::error!("Invalid number of arguments: {:?}", parts);
                        stream_write!(writer, "400\nInvalid number of arguments\n".to_string());
                        continue;
                    }

                    let rpc_id = parts[0].parse::<u64>();
                    let num_cores = parts[1].parse::<u32>();
                    let memory = parts[2].parse::<u64>();

                    if rpc_id.is_err() || num_cores.is_err() || memory.is_err() {
                        stream_write!(writer, "400\nInvalid arguments\n".to_string());
                        continue;
                    }

                    let rpc_id = rpc_id.unwrap();
                    let num_cores = num_cores.unwrap();
                    let memory = memory.unwrap();

                    log::info!("Changing resources for rpc_id: {}, num_cores: {}, memory: {}", rpc_id, num_cores, memory);

                    let ret = self.virt_server_manager.change_resources(rpc_id, num_cores, memory);
                    match ret {
                        Ok(_) => {
                            log::info!("Resources changed");
                            stream_write!(writer, "200\nDone\n".to_string());
                        }
                        Err(e) => {
                            log::error!("Error changing resources: {}", e);
                            stream_write!(writer, format!("500\n{}\n", e));
                        }
                    }
                }

                _ => {
                    log::error!("Unknown command: {}", buf);
                }
            }
        }
    }


}
