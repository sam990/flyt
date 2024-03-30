use std::{io::{BufRead, BufReader, Write}, net::TcpStream, sync::RwLock, thread};
use crate::common::utils::Utils;
use crate::common::api_commands::FlytApiCommand;
use crate::vcuda_client_handler::VCudaClientManager;

#[derive(Debug, Clone)]
pub struct VirtServer {
    pub address: String,
    pub rpc_id: u16
}

pub struct ResourceManagerHandler<'b> {
    server_ip: String,
    server_port: u16,
    stream: RwLock<Option<TcpStream>>,
    virt_server: RwLock<Option<VirtServer>>,
    client_mgr: &'b VCudaClientManager
}


impl <'b> ResourceManagerHandler <'b> {
    pub fn new(server_ip: String, server_port: u16, client_mgr: &'b VCudaClientManager) -> ResourceManagerHandler {

        ResourceManagerHandler {
            server_ip: server_ip,
            server_port: server_port,
            stream: RwLock::new(None),
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
                let vserver = ResourceManagerHandler::read_virt_server_details(&mut stream);
                if vserver.is_some() {
                    self.stream.write().unwrap().replace(stream);
                    self.virt_server.write().unwrap().replace(vserver.clone().unwrap());
                    self.launch_cmd_reader_thread(scope);
                    vserver
                }
                else {
                    None
                }

            }
            Err(error) => {
                println!("Error connecting to server: {}", error);
                None
            }
        }
    }


    fn read_virt_server_details(stream: &mut TcpStream) -> Option<VirtServer> {
        let mut stream_clone = stream.try_clone().unwrap();
        stream_clone.write_all(format!("{}\n", FlytApiCommand::CLIENTD_RMGR_CONNECT).as_bytes()).unwrap();
        let respone = Utils::read_response(stream_clone, 2);
        if respone[0] == "200" {
            let server_details = respone[1].split(",").collect::<Vec<&str>>();
            if server_details.len() != 2 {
                println!("Server details not in correct format: {}", respone[1]);
                return None;
            }
            Some(VirtServer {
                address: server_details[0].to_string(),
                rpc_id: server_details[1].parse::<u16>().unwrap()
            })
        } else {
            println!("Error getting server details: {} {}", respone[0], respone[1]);
            None
        }
    }

    fn change_virt_server(&self, response_str: String) {
        let server_details = response_str.split(",").collect::<Vec<&str>>();
        if server_details.len() != 2 {
            println!("Server details not in correct format: {}", response_str);
            return;
        }
        self.virt_server.write().unwrap().replace(VirtServer {
            address: server_details[0].to_string(),
            rpc_id: server_details[1].parse::<u16>().unwrap()
        });
    }

    pub fn notify_zero_clients(&self) -> bool {
        let mut stream = TcpStream::connect(format!("{}:{}", self.server_ip, self.server_port)).unwrap();
        stream.write_all(format!("{}\n", FlytApiCommand::CLIENTD_RMGR_ZERO_VCUDA_CLIENTS).as_bytes()).unwrap();
        let response = Utils::read_response(stream, 1);
        response[0] == "200"
    }


    fn launch_cmd_reader_thread<'a>(&'a self, scope: &'a thread::Scope<'a, '_>) {
        let stream_clone = self.stream.read().unwrap().as_ref().unwrap().try_clone().unwrap();
        scope.spawn( move || {
            let reader_clone = stream_clone.try_clone().unwrap();
            let mut writer = stream_clone;
            let mut reader = BufReader::new(reader_clone);

            loop {
                let mut command = String::new();
                let read_len = reader.read_line(&mut command).unwrap();
                
                if read_len == 0 {
                    println!("Connection closed by server");
                    break;
                }

                println!("Received command: {}", command);

                match command.trim() {
                    FlytApiCommand::RMGR_CLIENTD_PAUSE => {
                        let total_clients = self.client_mgr.num_active_clients();
                        let num_paused = self.client_mgr.pause_clients();
                        writer.write_all(format!("200\nPaused {} out of {} clients\n", num_paused, total_clients).as_bytes()).unwrap();
                    }

                    FlytApiCommand::RMGR_CLIENTD_RESUME => {
                        let num_resumed = self.client_mgr.resume_clients();
                        writer.write_all(format!("200\nResumed {} clients\n", num_resumed).as_bytes()).unwrap();
                    }

                    FlytApiCommand::RMGR_CLIENTD_CHANGE_VIRT_SERVER => {
                        let mut payload = String::new();
                        reader.read_line(&mut payload).unwrap();

                        self.change_virt_server(payload);
                        self.client_mgr.change_virt_server(self.virt_server.read().unwrap().as_ref().unwrap());
                        writer.write_all("200\nChanged virt server\n".as_bytes()).unwrap();
                    }
                    
                    FlytApiCommand::RMGR_CLIENTD_DEALLOC_VIRT_SERVER => {
                        let mut lock_guard = self.virt_server.write().unwrap();
                        let num_clients = self.client_mgr.num_active_clients();

                        if num_clients == 0 {
                            lock_guard.take();
                            writer.write_all("200\nDeallocated virt server\n".as_bytes()).unwrap();
                        } else {
                            writer.write_all(format!("500\n{} clients still active\n", num_clients).as_bytes()).unwrap();
                        }

                    }

                    _ => {
                        println!("Unknown command: {}", command);
                    }
                }
            }
        });
    }

}