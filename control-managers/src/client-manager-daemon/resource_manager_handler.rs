use std::{io::{BufRead, BufReader, Write}, net::TcpStream, sync::RwLock, thread};
use crate::common::utils::Utils;
use crate::common::api_commands::FlytApiCommand;

#[derive(Debug, Clone)]
pub struct VirtServer {
    pub address: String,
    pub rpc_id: u16
}

pub struct ResourceManagerHandler {
    server_ip: String,
    server_port: u16,
    stream: RwLock<Option<TcpStream>>,
    virt_server: RwLock<Option<VirtServer>>,
}


impl ResourceManagerHandler {
    pub fn new(server_ip: String, server_port: u16) -> ResourceManagerHandler {

        ResourceManagerHandler {
            server_ip: server_ip,
            server_port: server_port,
            stream: RwLock::new(None),
            virt_server: RwLock::new(None),
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

    fn say_hello(&self) -> String {
        "Hello".to_string()
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


    fn launch_cmd_reader_thread<'a>(&'a self, scope: &'a thread::Scope<'a, '_>) {
        let stream_clone = self.stream.read().unwrap().as_ref().unwrap().try_clone().unwrap();
        scope.spawn( move || {
            loop {
                let mut reader = BufReader::new(stream_clone.try_clone().unwrap());
                let mut command = String::new();
                reader.read_line(&mut command).unwrap();
                println!("Received command: {}", command);

                self.say_hello();
            }
        });
    }

}