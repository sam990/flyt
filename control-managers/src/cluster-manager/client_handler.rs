
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
    pub client_id: i32,
    pub stream: Arc<RwLock<Option<StreamEnds<TcpStream>>>>,
    pub virt_server: Option<Arc<RwLock<VirtServer>>>,
    pub is_migrating: RwLock<bool>,
    pub is_active: RwLock<bool>,
}

impl Clone for FlytClientNode {
    fn clone(&self) -> Self {
        FlytClientNode {
            ipaddr: self.ipaddr.clone(),
            client_id: self.client_id,
            stream: self.stream.clone(),
            virt_server: self.virt_server.clone(),
            is_migrating: RwLock::new(false),
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
    clients: Mutex<HashMap<(String, i32),FlytClientNode>>,
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
        clients.insert((client.ipaddr.to_string(), client.client_id), client);
    }

    pub fn update_stream(&self, ipaddr: &str, client_id: i32, writer: TcpStream, reader: BufReader<TcpStream>) {
        let mut clients = self.clients.lock().unwrap();
        let client = clients.get_mut(&(ipaddr.to_string(), client_id)).unwrap();

        client.stream.write().unwrap().replace(StreamEnds{writer, reader});
    }

    pub fn set_client_status(&self, ipaddr: &str, client_id: i32, status: bool) {
        let mut clients = self.clients.lock().unwrap();
        let client = clients.get_mut(&(ipaddr.to_string(), client_id));
        if let Some(client) = client {
            *client.is_active.get_mut().unwrap() = status;
        }
    }

    pub fn get_client_status(&self, ipaddr: &str, client_id: i32) -> bool {
        let clients = self.clients.lock().unwrap();
        let client = clients.get(&(ipaddr.to_string(), client_id));
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

    pub fn get_client(&self, ipaddr: &str, client_id: i32) -> Option<FlytClientNode> {
        let clients = self.clients.lock().unwrap();
        clients.get(&(ipaddr.to_string(), client_id)).cloned()
    }

    pub fn remove_client(&self, ipaddr: &str, client_id: i32) {
        let mut clients = self.clients.lock().unwrap();
        clients.remove(&(ipaddr.to_string(), client_id));
    }

    pub fn get_all_clients(&self) -> Vec<FlytClientNode> {
        let clients = self.clients.lock().unwrap();
        clients.values().cloned().collect()
    }

    pub fn exists(&self, ipaddr: &str, client_id: i32) -> bool {
        let clients = self.clients.lock().unwrap();
        clients.contains_key(&(ipaddr.to_string(), client_id ))
    }

    pub fn stop_client(&self, ipaddr: &str, client_id: i32) -> Result<(),String> {
        if self.exists(ipaddr, client_id) {
            let mut client = self.get_client(ipaddr, client_id).unwrap();
            *client.is_migrating.get_mut().unwrap() = true;
            if client.stream.read().unwrap().is_some() {
                let writer_resp = { get_writer!(client).write_all(format!("{},{}\n", FlytApiCommand::RMGR_CLIENTD_PAUSE, client_id).as_bytes()) };
                match writer_resp {
                    Ok(_) => {
                        let response = match StreamUtils::read_response(get_reader!(client), 2) {
                            Ok(response) => {
                                log::info!("Response from client {},{} for stop: {:?}", ipaddr, client_id, response);
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

    pub fn change_virt_server(&self, ipaddr: &str, client_id: i32, new_virt_server: &Arc<RwLock<VirtServer>>) -> Result<(), String> {
        if self.exists(ipaddr, client_id) {
            let mut client = self.get_client(ipaddr, client_id).unwrap();

            let new_snode_ip = new_virt_server.read().unwrap().ipaddr.clone();
            let new_rpc_id = new_virt_server.read().unwrap().rpc_id;

            if client.stream.read().unwrap().is_some() {
                let writer_resp = { get_writer!(client).write_all(format!("{},{}\n{},{}\n", FlytApiCommand::RMGR_CLIENTD_CHANGE_VIRT_SERVER, client_id, new_snode_ip, new_rpc_id).as_bytes()) };
                match writer_resp {
                    Ok(_) => {
                        let response = match StreamUtils::read_response(get_reader!(client), 2) {
                            Ok(response) => {
                                log::info!("Response from client {},{} for change virt server: {:?}", ipaddr, client_id, response);
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

    pub fn resume_client(&self, ipaddr: &str, client_id: i32) -> Result<(),String> {
        if self.exists(ipaddr, client_id) {
            let mut client = self.get_client(ipaddr, client_id).unwrap();
            *client.is_migrating.get_mut().unwrap() = false;
            if client.stream.read().unwrap().is_some() {
                let writer_resp = { get_writer!(client).write_all(format!("{},{}\n", FlytApiCommand::RMGR_CLIENTD_RESUME, client_id).as_bytes()) };
                match writer_resp {
                    Ok(_) => {
                        let response = match StreamUtils::read_response(get_reader!(client), 2) {
                            Ok(response) => {
                                log::info!("Response from client {},{} for resume: {:?}", ipaddr, client_id, response);
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

    fn get_client_command_client_id(response_str: String) -> (String, i32) {
        let client_details = response_str.split(",").collect::<Vec<&str>>();
        if client_details.len() != 2 {
            log::error!("client details not in correct format: {}", response_str);
            return ("".to_string(), -1);
        }
        (client_details[0].to_string(), client_details[1].trim().parse::<i32>().unwrap())
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
        
        let (command, client_id) = match StreamUtils::read_response(&mut reader, 1) {
            Ok(response) => Self::get_client_command_client_id(response[0].clone()),
            Err(e) => {
                log::error!("Error reading command from client: {}", e);
                return;
            }
        };
        
        match command.as_str() {
            FlytApiCommand::CLIENTD_RMGR_CONNECT => {
                log::info!("CLIENTD_RMGR_CONNECT command received from client {},{}", client_ip, client_id);

                /*
                if let Some(client) = self.get_client(&client_ip, client_id)  {
                        self.remove_client(&client_ip, client_id);
                        self.deallocate_vm_resources(&client_ip, client_id, client);
                }
                */
                if self.exists(&client_ip, client_id) && self.get_client_status(&client_ip, client_id) && self.get_client(&client_ip, client_id).unwrap().virt_server.is_some() {
                    self.set_client_status(&client_ip, client_id, true);
                    let client = self.get_client(&client_ip, client_id).unwrap();
                    let virt_server = client.virt_server.as_ref().unwrap().read().unwrap();
                    log::info!("reusing client {},{}", client_ip, client_id);
                    match stream.write_all(format!("200\n{},{}\n", virt_server.ipaddr, virt_server.rpc_id).as_bytes()) {
                        Ok(_) => {
                            self.update_stream(&client_ip, client_id, stream, reader);                  
                        }
                        Err(e) => {
                            log::error!("Error writing to stream: {}", e);
                        }
                    }
                }
                else {
                    log::info!("creating new client {},{}", client_ip, client_id);
                    let virt_server = self.server_nodes_manager.allocate_vm_resources(&client_ip, client_id);
                    log::info!("after allocating vm resources new client {},{}", client_ip, client_id);
                    if virt_server.is_ok() {
                        let virt_server = virt_server.unwrap();
                        let client = FlytClientNode {
                            ipaddr: client_ip.clone(),
                            client_id: client_id,
                            stream: Arc::new(RwLock::new(Some(StreamEnds{writer: stream, reader}))),
                            virt_server: Some(virt_server.clone()),
                            is_migrating: RwLock::new(false),
                            is_active: RwLock::new(true),
                        };
                        let client_stream_clone = client.stream.clone();
                        let mut client_stream_clone_writer = client_stream_clone.write().unwrap();
                        match client_stream_clone_writer.as_mut().unwrap().writer.write_all(format!("200\n{},{}\n", virt_server.read().unwrap().ipaddr, virt_server.read().unwrap().rpc_id).as_bytes()) {
                            Ok(_) => {
                                log::info!("Adding new client {},{}", client_ip, client_id);
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

            FlytApiCommand::CLIENTD_RMGR_DISCONNECT => {
                log::info!("CLIENTD_RMGR_DISCONNECT command received from client {},{}", client_ip, client_id);

                if let Some(mut client) = self.get_client(&client_ip, client_id)  {
                    if client.virt_server.is_none() {
                        log::info!("No virt server allocated for client to discoonect: {} {}", client_ip, client_id);
                        let _ = stream.write_all(format!("500\n").as_bytes());
                    }
                    else if *client.is_migrating.read().unwrap() {
                        log::info!("Client is migrating, cannot disconnect resources");
                        let _ = stream.write_all(format!("500\n").as_bytes());
                    }
                    else {
                        let virt_server_ip = client.virt_server.as_ref().unwrap().read().unwrap().ipaddr.clone();
                        let rpc_id = client.virt_server.as_ref().unwrap().read().unwrap().rpc_id;

                        self.server_nodes_manager.free_virt_server(&virt_server_ip, rpc_id, false);
                        let _ = stream.write_all(format!("200\n").as_bytes());

                        // Free the previous client stream as well
                        //self.update_stream(&client_ip, client_id, None, None);                  
                        client.virt_server = None;
                        self.remove_client(&client_ip, client_id);
                    }
                }
                else {
                    log::error!("CLIENTD_RMGR_DISCONNECT command received from non existing client {},{}", client_ip, client_id);
                    let _ = stream.write_all(format!("500\n").as_bytes());
                }
            },

            FlytApiCommand::CLIENTD_RMGR_ZERO_VCUDA_CLIENTS => {
                log::info!("CLIENTD_RMGR_ZERO_VCUDA_CLIENTS command received from client {},{}", client_ip, client_id);
                // deallocate all clients and servers...
                let clients = self.clients.lock().unwrap();
                let mut to_deallocate = Vec::new();

                for (key, client) in clients.iter() {
                    let ip = key.0.clone();
                    let cid = key.1.clone();
                    if ip == client_ip && self.get_client_status(&client_ip, cid) {
                        to_deallocate.push((ip, cid, client.clone()));
                    }
                }

                // Drop the lock on self.clients by releasing `clients`
                drop(clients);

                // Step 2: Iterate over the collected vector and perform deallocation
                for (ip, cid, client) in to_deallocate {
                    log::info!("deallocate for client ip: {} client {},{}", client_ip, ip, cid);

                    // Set the client status and remove the client outside the locked scope
                    self.set_client_status(&client_ip, cid, false);
                    self.remove_client(&client_ip, cid);

                    // Check for deallocation time and spawn a thread if needed
                    if let Some(dealloc_time) = get_virt_server_deallocate_time() {
                        let client_ip_for_thread = client_ip.clone();
                        scope.spawn(move || {
                            thread::sleep(std::time::Duration::from_secs(dealloc_time));
                            let _ = self.deallocate_vm_resources(&client_ip_for_thread, cid, client);
                        });
                    }
                }
                /*
                for (key, client) in clients.iter() {
                    let ip = key.0.clone();
                    let cid = key.1.clone();
                    log::info!("deallocate for client ip: {} client{},{}", client_ip, ip, cid);
                    if ip == client_ip {
                        if self.get_client_status(&client_ip, cid) {
                            self.set_client_status(&client_ip, cid, false);
                            self.remove_client(&client_ip, cid);
                            if let Some(dealloc_time) = get_virt_server_deallocate_time() {
                                let client_ip_for_thread = client_ip.clone();
                                scope.spawn(move || {
                                    thread::sleep(std::time::Duration::from_secs(dealloc_time));
                                    let _ = self.deallocate_vm_resources(&client_ip_for_thread, cid, client.clone());
                                });
                            }
                        }
                    }
                }
                */
                match stream.write_all("200\nDone\n".as_bytes()) {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("Error writing response to stream: {}", e);
                    }
                }
            },
            
            _ => {
                log::error!("Unknown command: {}", command);
            }
        }
    }

    fn deallocate_vm_resources(&self, ipaddr: &str, client_id: i32, mut client: FlytClientNode) -> Result<(),String> {
        log::info!("Deallocating virt server for client: {},{}", ipaddr, client_id);
        //let mut client = self.get_client(ipaddr, client_id).ok_or("Client not found".to_string())?;
        if client.virt_server.is_none() {
            log::info!("No virt server allocated for client: {} {}", ipaddr, client_id);
            return Err("No resources to deallocate".to_string());
        }

        if *client.is_active.read().unwrap() {
            log::info!("Client is active, cannot deallocate resources");
            return Err("Client is active".to_string());
        }
        if *client.is_migrating.read().unwrap() {
            log::info!("Client is migrating, cannot deallocate resources");
            return Err("Client is migrating".to_string());
        }


        if client.stream.read().unwrap().is_some() {
            log::trace!("Sending dealloc command to client: {} {}", ipaddr, client_id);
            match get_writer!(client).write_all(format!("{},{}\n", FlytApiCommand::RMGR_CLIENTD_DEALLOC_VIRT_SERVER, client_id).as_bytes()) {
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

        //let virt_server_ip = virt_server.read().unwrap().ipaddr.clone();
        let virt_server_ip = client.virt_server.as_ref().unwrap().read().unwrap().ipaddr.clone();
        let rpc_id = client.virt_server.as_ref().unwrap().read().unwrap().rpc_id;
        //let rpc_id = virt_server.read().unwrap().rpc_id;
        //drop(lock_guard);
        self.server_nodes_manager.free_virt_server(&virt_server_ip, rpc_id, true)?;

        client.virt_server = None;

        //self.update_client(client);

        Ok(())
    }
}

