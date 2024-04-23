use std::{io::{BufRead, BufReader, Write}, net::TcpStream, sync::RwLock, thread};
use crate::common::utils::StreamUtils;
use crate::common::api_commands::FlytApiCommand;
use crate::vcuda_client_handler::VCudaClientManager;

#[derive(Debug, Clone)]
pub struct VirtServer {
    pub address: String,
    pub rpc_id: u64,
}

pub struct ResourceManagerHandler<'b> {
    server_ip: String,
    server_port: u16,
    virt_server: RwLock<Option<VirtServer>>,
    client_mgr: &'b VCudaClientManager
}


impl <'b> ResourceManagerHandler <'b> {
    pub fn new(server_ip: String, server_port: u16, client_mgr: &'b VCudaClientManager) -> ResourceManagerHandler {

        ResourceManagerHandler {
            server_ip: server_ip,
            server_port: server_port,
            virt_server: RwLock::new(None),
            client_mgr: client_mgr
        }
    }

    pub fn get_virt_server<'a>(&'a self, scope: &'a thread::Scope<'a,'_>) -> Option<VirtServer> {
        if self.virt_server.read().unwrap().is_some() {
            return self.virt_server.read().unwrap().clone();
        }
        let stream = TcpStream::connect(format!("{}:{}", self.server_ip, self.server_port));
        match stream {
            Ok(mut stream) => {
                let stream_clone = match stream.try_clone() {
                    Ok(stream) => stream,
                    Err(e) => {
                        log::error!("Error cloning stream: {}", e);
                        return None;
                    }
                };
                let mut reader = BufReader::new(stream_clone);
                let vserver = ResourceManagerHandler::read_virt_server_details(&mut reader, &mut stream);
                if vserver.is_some() {
                    self.virt_server.write().unwrap().replace(vserver.clone().unwrap());
                    self.launch_cmd_reader_thread(scope, reader, stream);
                    vserver
                }
                else {
                    None
                }

            }
            Err(error) => {
                log::error!("Error connecting to server: {}", error);
                None
            }
        }
    }

    pub fn virt_server_available(&self) -> bool {
        self.virt_server.read().unwrap().is_some()
    }

    fn read_virt_server_details(reader: &mut BufReader<TcpStream>, stream: &mut TcpStream) -> Option<VirtServer> {
        match stream.write_all(format!("{}\n", FlytApiCommand::CLIENTD_RMGR_CONNECT).as_bytes()) {
            Ok(_) => {}
            Err(e) => {
                log::error!("Error writing to stream: {}", e);
                return None;
            }
        }
        
        let response =  match StreamUtils::read_response(reader, 2) {
            Ok(response) => {
                response
            }
            Err(error) => {
                log::error!("Error reading response: {}", error);
                return  None;
            }
        };

        if response[0] == "200" {
            let server_details = response[1].split(",").collect::<Vec<&str>>();
            if server_details.len() != 2 {
                log::error!("Server details not in correct format: {}", response[1]);
                return None;
            }
            Some(VirtServer {
                address: server_details[0].to_string(),
                rpc_id: server_details[1].parse::<u64>().unwrap()
            })
        } else {
            log::error!("Error getting server details: {} {}", response[0], response[1]);
            None
        }
    }

    fn change_virt_server(&self, response_str: String) {
        let server_details = response_str.split(",").collect::<Vec<&str>>();
        if server_details.len() != 2 {
            log::error!("Server details not in correct format: {}", response_str);
            return;
        }
        self.virt_server.write().unwrap().replace(VirtServer {
            address: server_details[0].to_string(),
            rpc_id: server_details[1].parse::<u64>().unwrap()
        });
    }

    pub fn notify_zero_clients(&self) -> bool {
        let mut stream = match TcpStream::connect(format!("{}:{}", self.server_ip, self.server_port)) {
            Ok(stream) => {
                stream
            }
            Err(error) => {
                log::error!("Error connecting to server: {}", error);
                return false;
            }
        };
        match stream.write_all(format!("{}\n", FlytApiCommand::CLIENTD_RMGR_ZERO_VCUDA_CLIENTS).as_bytes()) {
            Ok(_) => {}
            Err(e) => {
                log::error!("Error writing to stream: {}", e);
                return false;
            }
        };
        let mut reader = BufReader::new(stream);
        let response = match StreamUtils::read_response(&mut reader, 2) {
            Ok(response) => {
                response
            }
            Err(error) => {
                log::error!("Error reading response: {}", error);
                return  false;
            }
        };
        response[0] == "200"
    }


    fn launch_cmd_reader_thread<'a>(&'a self, scope: &'a thread::Scope<'a, '_>, mut reader: BufReader<TcpStream>, mut writer: TcpStream) {
        scope.spawn( move || {

            loop {
                let mut command = String::new();
                let read_len = match reader.read_line(&mut command) {
                    Ok(read_len) => {
                        read_len
                    }
                    Err(error) => {
                        log::error!("Error reading command: {}", error);
                        break;
                    }
                };
                
                if read_len == 0 {
                    log::info!("Connection closed by server");
                    log::info!("Clearing Virt server");
                    self.virt_server.write().unwrap().take();
                    break;
                }

                log::info!("Received command: {}", command);

                match command.trim() {
                    FlytApiCommand::RMGR_CLIENTD_PAUSE => {
                        let total_clients = self.client_mgr.num_active_clients();
                        let num_paused = self.client_mgr.pause_clients();
                        let _ = StreamUtils::write_all(&mut writer, format!("200\nPaused {} out of {} clients\n", num_paused, total_clients));
                    }

                    FlytApiCommand::RMGR_CLIENTD_RESUME => {
                        let num_resumed = self.client_mgr.resume_clients();
                        let _ = StreamUtils::write_all(&mut writer, format!("200\nResumed {} clients\n", num_resumed));
                    }

                    FlytApiCommand::RMGR_CLIENTD_CHANGE_VIRT_SERVER => {
                        let payload = match StreamUtils::read_line(&mut reader) {
                            Ok(payload) => {
                                payload
                            }
                            Err(error) => {
                                log::error!("Error reading new virt servers. Skipping: {}", error);
                                continue;
                            }
                        };
                        self.change_virt_server(payload);
                        self.client_mgr.change_virt_server(self.virt_server.read().unwrap().as_ref().unwrap());
                        let _ = StreamUtils::write_all(&mut writer, "200\nChanged virt server\n".to_string());
                    }
                    
                    FlytApiCommand::RMGR_CLIENTD_DEALLOC_VIRT_SERVER => {
                        log::info!("Received deallocate virt server command");
                        let mut lock_guard = self.virt_server.write().unwrap();
                        let num_clients = self.client_mgr.num_active_clients();

                        if num_clients == 0 {
                            lock_guard.take();
                            log::info!("Deallocated virt server: {:?}", lock_guard.as_ref());
                            let _ = StreamUtils::write_all(&mut writer, "200\nDeallocated virt server\n".to_string());
                        } else {
                            log::info!("Cannot deallocate virt server, {} clients still active", num_clients);
                            let _ = StreamUtils::write_all(&mut writer, format!("500\n{} clients still active\n", num_clients));
                        }

                    }

                    _ => {
                        log::error!("Unknown command: {}", command);
                    }
                }
            }
        });
    }

}