
use crate::bookkeeping::*;
use crate::common::types::StreamEnds;
use crate::servernode_handler::ServerNodesManager;
use crate::common::api_commands::FlytApiCommand;
use crate::common::utils::StreamUtils;


use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

#[derive(Debug)]
pub struct FlytClientNode {
    pub ipaddr: String,
    pub stream: Arc<RwLock<Option<StreamEnds<TcpStream>>>>,
    pub virt_server: Option<Arc<RwLock<VirtServer>>>,
    pub is_active: RwLock<bool>,
}

impl Clone for FlytClientNode {
    fn clone(&self) -> Self {
        FlytClientNode {
            ipaddr: self.ipaddr.clone(),
            stream: self.stream.clone(),
            virt_server: self.virt_server.clone(),
            is_active: RwLock::new(*self.is_active.read().unwrap()),
        }
    }
}

macro_rules! get_reader {
    ($node: expr) => {
        $node.stream.write().unwrap().as_mut().unwrap().reader.by_ref()
    };
}

macro_rules! get_writer {
    ($node: expr) => {
        $node.stream.write().unwrap().as_mut().unwrap().writer
    };
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

    pub fn update_stream(&self, ipaddr: &str, writer: TcpStream, reader: BufReader<TcpStream>) {
        let mut clients = self.clients.lock().unwrap();
        let client = clients.get_mut(ipaddr).unwrap();

        client.stream.write().unwrap().replace(StreamEnds{writer, reader});
    }

    pub fn set_client_status(&self, ipaddr: &str, status: bool) {
        let mut clients = self.clients.lock().unwrap();
        let client = clients.get_mut(ipaddr);
        if let Some(client) = client {
            *client.is_active.get_mut().unwrap() = status;
        }
    }

    pub fn get_client_status(&self, ipaddr: &str) -> bool {
        let clients = self.clients.lock().unwrap();
        let client = clients.get(ipaddr);
        if let Some(client) = client {
            *client.is_active.read().unwrap()
        }
        else {
            false
        }
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

    pub fn stop_client(&self, ipaddr: &str) -> Result<(),String> {
        if self.exists(ipaddr) {
            let client = self.get_client(ipaddr).unwrap();
            if client.stream.read().unwrap().is_some() {
                let writer_resp = { get_writer!(client).write_all(format!("{}\n", FlytApiCommand::RMGR_CLIENTD_PAUSE).as_bytes()) };
                match writer_resp {
                    Ok(_) => {
                        let response = match StreamUtils::read_response(get_reader!(client), 2) {
                            Ok(response) => {
                                log::info!("Response from client {} for stop: {:?}", ipaddr, response);
                                response
                            }
                            Err(e) => {
                                log::error!("Error reading response from client: {}", e);
                                return Err("Error reading response from client".to_string());
                            }
                        };
                        if response[0] != "200" {
                            return Err(format!("Error stopping client: {}", response[1]));
                        }

                        Ok(())

                    }
                    Err(e) => {
                        log::error!("Error writing to client: {}", e);
                        return Err("Error writing to client".to_string());
                    }
                }
            }
            else {
                log::info!("Client stream is None");
                Err("Client stream is None".to_string())
            }
        }
        else {
            Err("Client not found".to_string())
        }
    }

    pub fn change_virt_server(&self, ipaddr: &str, new_virt_server: &Arc<RwLock<VirtServer>>) -> Result<(), String> {
        if self.exists(ipaddr) {
            let mut client = self.get_client(ipaddr).unwrap();

            let new_snode_ip = new_virt_server.read().unwrap().ipaddr.clone();
            let new_rpc_id = new_virt_server.read().unwrap().rpc_id;

            if client.stream.read().unwrap().is_some() {
                let writer_resp = { get_writer!(client).write_all(format!("{}\n{},{}\n", FlytApiCommand::RMGR_CLIENTD_CHANGE_VIRT_SERVER, new_snode_ip, new_rpc_id).as_bytes()) };
                match writer_resp {
                    Ok(_) => {
                        let response = match StreamUtils::read_response(get_reader!(client), 2) {
                            Ok(response) => {
                                log::info!("Response from client {} for change virt server: {:?}", ipaddr, response);
                                response
                            }
                            Err(e) => {
                                log::error!("Error reading response from client: {}", e);
                                return Err("Error reading response from client".to_string());
                            }
                        };
                        if response[0] != "200" {
                            return Err(format!("Error changing virt server: {}", response[1]));
                        }
                        
                        // Update client virt server
                        client.virt_server = Some(new_virt_server.clone());
                        self.update_client(client);
    
                        Ok(())

                    }
                    Err(e) => {
                        log::error!("Error writing to client: {}", e);
                        return Err("Error writing to client".to_string());
                    }
                }
            }
            else {
                log::info!("Client stream is None");
                Err("Client stream is None".to_string())
            }
        }
        else {
            Err("Client not found".to_string())
        }
    }

    pub fn resume_client(&self, ipaddr: &str) -> Result<(),String> {
        if self.exists(ipaddr) {
            let client = self.get_client(ipaddr).unwrap();
            if client.stream.read().unwrap().is_some() {
                let writer_resp = { get_writer!(client).write_all(format!("{}\n", FlytApiCommand::RMGR_CLIENTD_RESUME).as_bytes()) };
                match writer_resp {
                    Ok(_) => {
                        let response = match StreamUtils::read_response(get_reader!(client), 2) {
                            Ok(response) => {
                                log::info!("Response from client {} for resume: {:?}", ipaddr, response);
                                response
                            }
                            Err(e) => {
                                log::error!("Error reading response from client: {}", e);
                                return Err("Error reading response from client".to_string());
                            }
                        };
                        if response[0] != "200" {
                            return Err(format!("Error resuming client: {}", response[1]));
                        }

                        Ok(())

                    }
                    Err(e) => {
                        log::error!("Error writing to client: {}", e);
                        return Err("Error writing to client".to_string());
                    }
                }
            }
            else {
                log::info!("Client stream is None");
                Err("Client stream is None".to_string())
            }
        }
        else {
            Err("Client not found".to_string())
        }
    }

    pub fn start_flytclient_handler<'b>(&'b self, port: u16, scope: &'b thread::Scope<'b, '_>) {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    self.handle_flytclient(stream, scope)
                }
                Err(e) => {
                    log::error!("Error: {}", e);
                }
            }
        }
    }

    fn handle_flytclient<'b>(&'b self, mut stream: TcpStream, scope: &'b thread::Scope<'b, '_>) {
        let client_ip = stream.peer_addr().unwrap().ip().to_string();

        let stream_clone = match stream.try_clone() {
            Ok(stream) => stream,
            Err(e) => {
                log::error!("Error cloning stream: {}", e);
                return;
            }
        };

        let mut reader = BufReader::new(stream_clone);
        
        let command = match StreamUtils::read_response(&mut reader, 1) {
            Ok(command) => command,
            Err(e) => {
                log::error!("Error reading command from client: {}", e);
                return;
            }
        };
        
        match command[0].as_str() {
            FlytApiCommand::CLIENTD_RMGR_CONNECT => {
                log::info!("CLIENTD_RMGR_CONNECT command received from client {}", client_ip);

                if self.exists(&client_ip) && self.get_client(&client_ip).unwrap().virt_server.is_some() {
                    self.set_client_status(&client_ip, true);
                    let client = self.get_client(&client_ip).unwrap();
                    let virt_server = client.virt_server.as_ref().unwrap().read().unwrap();
                    match stream.write_all(format!("200\n{},{}\n", virt_server.ipaddr, virt_server.rpc_id).as_bytes()) {
                        Ok(_) => {
                            self.update_stream(&client_ip, stream, reader);                  
                        }
                        Err(e) => {
                            log::error!("Error writing to stream: {}", e);
                        }
                    }
                }
                else {
                    let virt_server = self.server_nodes_manager.allocate_vm_resources(&client_ip);
                    if virt_server.is_ok() {
                        let virt_server = virt_server.unwrap();
                        let client = FlytClientNode {
                            ipaddr: client_ip.clone(),
                            stream: Arc::new(RwLock::new(Some(StreamEnds{writer: stream, reader}))),
                            virt_server: Some(virt_server.clone()),
                            is_active: RwLock::new(true),
                        };
                        let client_stream_clone = client.stream.clone();
                        let mut client_stream_clone_writer = client_stream_clone.write().unwrap();
                        match client_stream_clone_writer.as_mut().unwrap().writer.write_all(format!("200\n{},{}\n", virt_server.read().unwrap().ipaddr, virt_server.read().unwrap().rpc_id).as_bytes()) {
                            Ok(_) => {
                                log::info!("Adding new client {}", client_ip);
                                self.add_client(client);
                            }
                            Err(e) => {
                                log::error!("Error writing to stream: {}", e);
                            }
                        }
                    }
                    else {
                        let _ = stream.write_all(format!("500\n{}\n", virt_server.unwrap_err()).as_bytes());
                    }
                }
            },

            FlytApiCommand::CLIENTD_RMGR_ZERO_VCUDA_CLIENTS => {
                log::info!("CLIENTD_RMGR_ZERO_VCUDA_CLIENTS command received from client {}", client_ip);
                if self.get_client_status(&client_ip) {
                    self.set_client_status(&client_ip, false);
                    if let Some(dealloc_time) = get_virt_server_deallocate_time() {
                        scope.spawn(move || {
                            thread::sleep(std::time::Duration::from_secs(dealloc_time));
                            let _ = self.deallocate_vm_resources(&client_ip);
                        });
                    }
                }
                match stream.write_all("200\nDone\n".as_bytes()) {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("Error writing response to stream: {}", e);
                    }
                }
            },
            
            _ => {
                log::error!("Unknown command: {}", command[0]);
            }
        }
        
    }

    fn deallocate_vm_resources(&self, ipaddr: &str) -> Result<(),String> {
        log::info!("Deallocating virt server for client: {}", ipaddr);
        let mut client = self.get_client(ipaddr).ok_or("Client not found".to_string())?;
        if client.virt_server.is_none() {
            log::info!("No virt server allocated for client: {}", ipaddr);
            return Err("No resources to deallocate".to_string());
        }

        let virt_server = client.virt_server.clone().unwrap(); 

        let lock_guard = virt_server.write().unwrap();
        if *client.is_active.read().unwrap() {
            log::info!("Client is active, cannot deallocate resources");
            return Err("Client is active".to_string());
        }


        if client.stream.read().unwrap().is_some() {
            log::trace!("Sending dealloc command to client: {}", ipaddr);
            match get_writer!(client).write_all(format!("{}\n", FlytApiCommand::RMGR_CLIENTD_DEALLOC_VIRT_SERVER).as_bytes()) {
                Ok(_) => {}
                Err(e) => {
                    log::error!("Error writing to client: {}", e);
                    return Err("Error writing to client".to_string());
                }
            };
            let response = match StreamUtils::read_response(get_reader!(client), 2) {
                Ok(response) => response,
                Err(e) => {
                    log::error!("Error reading response from client: {}", e);
                    return Err("Error reading response from client".to_string());
                }
            };
            log::info!("Response from client {} for deallocate: {:?}", ipaddr, response);
            if response[0] != "200" {
                return Err("Error deallocating resources".to_string());
            }
        }

        let virt_server_ip = lock_guard.ipaddr.clone();
        let rpc_id = lock_guard.rpc_id;
        drop(lock_guard);
        self.server_nodes_manager.free_virt_server(&virt_server_ip, rpc_id)?;

        client.virt_server = None;

        self.update_client(client);

        Ok(())
    }

    
}

