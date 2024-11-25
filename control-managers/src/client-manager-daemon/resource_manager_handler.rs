use std::{io::{BufRead, BufReader, Write}, net::TcpStream, sync::RwLock, thread};
use crate::common::utils::StreamUtils;
use crate::common::api_commands::FlytApiCommand;
use crate::vcuda_client_handler::VCudaClientManager;

pub struct ResourceManagerHandler<'b> {
    server_ip: String,
    server_port: u16,
    client_mgr: &'b VCudaClientManager
}


impl <'b> ResourceManagerHandler <'b> {
    pub fn new(server_ip: String, server_port: u16, client_mgr: &'b VCudaClientManager) -> ResourceManagerHandler {

        ResourceManagerHandler {
            server_ip: server_ip,
            server_port: server_port,
            client_mgr: client_mgr
        }
    }

    pub fn get_virt_server<'a>(&'a self, scope: &'a thread::Scope<'a,'_>, gid: i32, sm_core: i32, rmgrflag: bool) -> Option<(String, u64)> {
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
                if rmgrflag == true {
                    match stream.write_all(format!("{},{},{}\n", FlytApiCommand::CLIENTD_RMGR_CONNECT, gid, sm_core).as_bytes()) {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Error writing to stream: {}", e);
                            return None;
                        }
                    }
                    
                    let response =  match StreamUtils::read_response(&mut reader, 2) {
                        Ok(response) => {
                            response
                        }
                        Err(error) => {
                            log::error!("Error reading response: {}", error);
                            return  None;
                        }
                    };

                    log::debug!("Cluster manager response for connect: {:?}", response);

                    if response[0] == "200" {
                        let server_details = response[1].split(",").collect::<Vec<&str>>();
                        if server_details.len() != 2 {
                            log::error!("Server details not in correct format: {}", response[1]);
                            return None;
                        }
                        self.launch_cmd_reader_thread(scope, reader, stream);
                        Some((
                            server_details[0].to_string(),
                            server_details[1].parse::<u64>().unwrap()
                        ))
                    } else {
                        log::error!("Error getting server details: {} {}", response[0], response[1]);
                        None
                    }
                }
                else {
                    // Disconnect command.
                    match stream.write_all(format!("{},{}\n", FlytApiCommand::CLIENTD_RMGR_DISCONNECT, gid).as_bytes()) {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Error writing to stream: {}", e);
                            return None;
                        }
                    }
                    
                    let response =  match StreamUtils::read_response(&mut reader, 1) {
                        Ok(response) => {
                            response
                        }
                        Err(error) => {
                            log::error!("Error reading response: {}", error);
                            return  None;
                        }
                    };

                    log::debug!("Cluster manager response for disconnect: {:?}", response);

                    if response[0] != "200" {
                        log::error!("Error disconnectiving server details: {}", response[0]);
                    }
                    //self.launch_cmd_reader_thread(scope, reader, stream);
                    None
                }
            }
            Err(error) => {
                log::error!("Error connecting to server: {}", error);
                None
            }
        }
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
        match stream.write_all(format!("{},{}\n", FlytApiCommand::CLIENTD_RMGR_ZERO_VCUDA_CLIENTS, -1).as_bytes()) {
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

    fn get_client_command_gid(response_str: String) -> Option<(String, i32)> {
        let client_details = response_str.split(",").collect::<Vec<&str>>();
        if client_details.len() != 2 {
            log::error!("client details not in correct format: {}", response_str);
            None
        }
        else {
            log::info!("get_client_command_gid {}, part 1 :{} part 2: {}", response_str, client_details[0], client_details[1]);
            Some((client_details[0].to_string(), client_details[1].trim().parse::<i32>().unwrap()))
        }
    }

    fn launch_cmd_reader_thread<'a>(&'a self, scope: &'a thread::Scope<'a, '_>, mut reader: BufReader<TcpStream>, mut writer: TcpStream) {
        scope.spawn( move || {

            loop {
                let mut command_str = String::new();
                let read_len = match reader.read_line(&mut command_str) {
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
                    //self.virt_server.write().unwrap().take();
                    break;
                }

                let Some((command, gid)) = ResourceManagerHandler::get_client_command_gid(command_str) else { todo!() };
                log::info!("Received command: {}, {}", command, gid);

                match command.trim() {
                    FlytApiCommand::RMGR_CLIENTD_PAUSE => {
                        let num_paused = self.client_mgr.pause_client(gid);
                        let ret = StreamUtils::write_all(&mut writer, format!("200\nPaused client id{} \n", gid));
                        match ret {
                            Ok(_) => {
                                log::info!("Successfully wrote to stream");
                            }
                            Err(e) => {
                                log::error!("Error writing to stream: {}", e);
                            }
                            
                        }
                    }

                    FlytApiCommand::RMGR_CLIENTD_RESUME => {
                        let num_resumed = self.client_mgr.resume_client(gid);
                        let _ = StreamUtils::write_all(&mut writer, format!("200\nResumed {} clients\n", num_resumed));
                    }

                    FlytApiCommand::RMGR_CLIENTD_CHANGE_VIRT_SERVER => {
                        let response_str = match StreamUtils::read_line(&mut reader) {
                            Ok(response_str) => {
                                response_str
                            }
                            Err(error) => {
                                log::error!("Error reading new virt servers. Skipping: {}", error);
                                continue;
                            }
                        };
                        let server_details = response_str.split(",").collect::<Vec<&str>>();
                        if server_details.len() != 2 {
                            log::error!("Server details not in correct format: {}", response_str);
                            return;
                        }
                        let address = server_details[0].to_string();
                        let rpc_id =  server_details[1].parse::<u64>().unwrap();

                        self.client_mgr.change_virt_server(address, rpc_id, gid);
                        let _ = StreamUtils::write_all(&mut writer, "200\nChanged virt server\n".to_string());
                    }
                    
                    FlytApiCommand::RMGR_CLIENTD_DEALLOC_VIRT_SERVER => {
                        log::info!("Received deallocate virt server command");
                        //TODO: What is this doing, make it gid specific
                        /*
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
                        */

                    }

                    _ => {
                        log::error!("Unknown command: {}", command);
                    }
                }
            }
        });
    }

}
