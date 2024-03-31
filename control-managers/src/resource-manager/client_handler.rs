
use crate::bookkeeping::*;
use crate::servernode_handler::ServerNodesManager;
use crate::common::api_commands::FlytApiCommand;
use crate::common::utils::Utils;


use std::collections::HashMap;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::{Mutex, RwLock};
use std::thread;


#[derive(Debug)]
pub struct FlytClientNode {
    pub ipaddr: String,
    pub stream: Option<TcpStream>,
    pub virt_server: RwLock<Option<VirtServer>>,
    pub is_active: RwLock<bool>,
}

impl Clone for FlytClientNode {
    fn clone(&self) -> Self {
        FlytClientNode {
            ipaddr: self.ipaddr.clone(),
            stream: self.stream.as_ref().map(|stream| stream.try_clone().unwrap()),
            virt_server: self.virt_server.read().unwrap().clone().into(),
            is_active: RwLock::new(*self.is_active.read().unwrap()),
        }
    }
}

pub struct FlytClientManager<'a> {
    clients: Mutex<HashMap<String,FlytClientNode>>,
    server_nodes_manager: &'a ServerNodesManager<'a>,
}


impl<'a> FlytClientManager<'a> {
    
    pub fn new(server_nodes_mgr: &'a ServerNodesManager) -> Self {
        FlytClientManager {
            clients: Mutex::new(HashMap::new()),
            server_nodes_manager: server_nodes_mgr,
        }
    }

    pub fn add_client(&self, client: FlytClientNode) {
        let mut clients = self.clients.lock().unwrap();
        clients.insert(client.ipaddr.clone(), client);
    }

    pub fn update_stream(&self, ipaddr: &str, stream: TcpStream) {
        let mut clients = self.clients.lock().unwrap();
        let client = clients.get_mut(ipaddr).unwrap();
        client.stream.replace(stream);
    }

    pub fn set_client_status(&self, ipaddr: &str, status: bool) {
        let mut clients = self.clients.lock().unwrap();
        let client = clients.get_mut(ipaddr).unwrap();
        *client.is_active.write().unwrap() = status;
    }

    pub fn update_client(&self, client: FlytClientNode) {
        self.add_client(client);
    }

    pub fn get_client(&self, ipaddr: &str) -> Option<FlytClientNode> {
        let clients = self.clients.lock().unwrap();
        clients.get(ipaddr).cloned()
    }

    pub fn remove_client(&self, ipaddr: &str) {
        let mut clients = self.clients.lock().unwrap();
        clients.remove(ipaddr);
    }

    pub fn get_all_clients(&self) -> Vec<FlytClientNode> {
        let clients = self.clients.lock().unwrap();
        clients.values().cloned().collect()
    }

    pub fn exists(&self, ipaddr: &str) -> bool {
        let clients = self.clients.lock().unwrap();
        clients.contains_key(ipaddr)
    }


    pub fn start_flytclient_handler<'b>(&'b self, port: u16, scope: &'b thread::Scope<'b, '_>) {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    self.handle_flytclient(stream, scope)
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
        }
    }

    fn handle_flytclient<'b>(&'b self, mut stream: TcpStream, scope: &'b thread::Scope<'b, '_>) {
        let client_ip = stream.peer_addr().unwrap().ip().to_string();
        
        let command = Utils::read_response(&mut stream, 1);
        
        match command[0].as_str() {
            FlytApiCommand::CLIENTD_RMGR_CONNECT => {
                if self.exists(&client_ip) && self.get_client(&client_ip).unwrap().virt_server.read().unwrap().is_some() {
                    self.set_client_status(&client_ip, true);
                    let virt_server = self.get_client(&client_ip).unwrap().virt_server.read().unwrap().clone().unwrap();
                    stream.write_all(format!("200\n{},{}\n", virt_server.ipaddr, virt_server.rpc_id).as_bytes()).unwrap();
                    self.update_stream(&client_ip, stream);                  
                }
                else {
                    let virt_server = self.server_nodes_manager.allocate_vm_resources(&client_ip);
                    if virt_server.is_ok() {
                        let virt_server = virt_server.unwrap();
                        let client = FlytClientNode {
                            ipaddr: client_ip.clone(),
                            stream: Some(stream.try_clone().unwrap()),
                            virt_server: RwLock::new(Some(virt_server.clone())),
                            is_active: RwLock::new(true),
                        };
                        self.add_client(client);
                        stream.write_all(format!("200\n{},{}\n", virt_server.ipaddr, virt_server.rpc_id).as_bytes()).unwrap();
                    }
                    else {
                        stream.write_all(format!("500\n{}\n", virt_server.unwrap_err()).as_bytes()).unwrap();
                        if let Some(dealloc_time) = get_virt_server_deallocate_time() {
                            scope.spawn(move || {
                                thread::sleep(std::time::Duration::from_secs(dealloc_time));
                                let _ = self.deallocate_vm_resources(&client_ip);
                            });
                        }
                    }
                }
            },

            FlytApiCommand::CLIENTD_RMGR_ZERO_VCUDA_CLIENTS => {
                self.set_client_status(&client_ip, false);
                stream.write_all("200\nDone\n".as_bytes()).unwrap();
            },
            
            _ => {
                println!("Unknown command: {}", command[0]);
            }
        }
        
    }

    fn deallocate_vm_resources(&self, ipaddr: &str) -> Result<(),String> {
        let mut client = self.get_client(ipaddr).ok_or("Client not found".to_string())?;
        let mut lock_guard = client.virt_server.write().unwrap();
        if *client.is_active.read().unwrap() {
            return Err("Client is active".to_string());
        }

        if lock_guard.is_none() {
            return Err("No resources to deallocate".to_string());
        }

        if client.stream.is_some() {
            client.stream.as_ref().unwrap().write_all(format!("{}\n", FlytApiCommand::RMGR_CLIENTD_DEALLOC_VIRT_SERVER).as_bytes()).unwrap();
            let response = Utils::read_response(client.stream.as_mut().unwrap(), 2);
            if response[0] != "200" {
                return Err("Error deallocating resources".to_string());
            }
        }

        let virt_server_ip = lock_guard.as_ref().unwrap().ipaddr.clone();
        let rpc_id = lock_guard.as_ref().unwrap().rpc_id;
        self.server_nodes_manager.free_virt_server(virt_server_ip, rpc_id)?;

        lock_guard.take();

        Ok(())
    }

    
}

