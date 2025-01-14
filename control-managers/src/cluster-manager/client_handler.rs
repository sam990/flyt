
use crate::bookkeeping::*;
use crate::common::types::StreamEnds;
use crate::servernode_handler::ServerNodesManager;
use crate::common::api_commands::FlytApiCommand;
use crate::common::utils::StreamUtils;
use crate::common::server_metrics::{ServerMetricsInfo, ClientMetricsInfo};


use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use uuid::Uuid;

#[derive(Debug)]
pub struct FlytClientNode {
    pub ipaddr: String,
    pub client_id: i32,
    pub group_id: i32,
    pub stream: Arc<RwLock<Option<StreamEnds<TcpStream>>>>,
    pub virt_server: Option<Arc<RwLock<VirtServer>>>,
    pub is_migrating: RwLock<bool>,
    pub is_active: RwLock<bool>,
    pub compute_requested: u32,
    pub memory_requested: u64,
    pub server_metrics: Vec<ServerMetricsInfo>,
    pub client_metrics: Vec<ClientMetricsInfo>,
}

impl Clone for FlytClientNode {
    fn clone(&self) -> Self {
        FlytClientNode {
            ipaddr: self.ipaddr.clone(),
            client_id: self.client_id,
            group_id:self.group_id,
            stream: self.stream.clone(),
            virt_server: self.virt_server.clone(),
            is_migrating: RwLock::new(false),
            is_active: RwLock::new(*self.is_active.read().unwrap()),
            compute_requested: self.compute_requested,
            memory_requested: self.memory_requested,
            server_metrics: self.server_metrics.clone(),
            client_metrics: self.client_metrics.clone(),
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
    ip_grouping_enabled: Mutex<HashMap<String, bool>>,  // Enable/disable grouping per IP
    ipaddr_to_group_id: Mutex<HashMap<String, i32>>,    // Tracks group IDs for IPs with grouping enabled
    group_to_virt_server: Mutex<HashMap<i32, Arc<RwLock<VirtServer>>>>,  // Shared virt_server per group
}


impl<'a> FlytClientManager<'a> {
    
    pub fn new(server_nodes_mgr: &'a ServerNodesManager) -> Self {
        FlytClientManager {
            clients: Mutex::new(HashMap::new()),
            server_nodes_manager: server_nodes_mgr,
            ip_grouping_enabled: Mutex::new(HashMap::new()),
            ipaddr_to_group_id: Mutex::new(HashMap::new()),
            group_to_virt_server: Mutex::new(HashMap::new()),
        }
    }

    /// Enable or disable grouping for a specific IP
    pub fn set_grouping_for_ip(&self, ipaddr: String, enable_grouping: bool) {
        let mut ip_grouping = self.ip_grouping_enabled.lock().unwrap();
        ip_grouping.insert(ipaddr, enable_grouping);
    }

    /// Retrieves the virt_server for a given ipaddr.
    pub fn get_virt_server_for_ip(&self, ipaddr: &str) -> Option<Arc<RwLock<VirtServer>>> {
        let grouping_enabled = {
            let ip_grouping = self.ip_grouping_enabled.lock().unwrap();
            *ip_grouping.get(ipaddr).unwrap_or(&false)
        };

        if grouping_enabled {
            let ip_to_group = self.ipaddr_to_group_id.lock().unwrap();
            if let Some(group_id) = ip_to_group.get(ipaddr) {
                let group_to_virt = self.group_to_virt_server.lock().unwrap();
                return group_to_virt.get(group_id).cloned();
            }
        }

        None
    }

    pub fn count_clients_with_virt_server(&self, virt_server: Arc<RwLock<VirtServer>>) -> usize {
        // Lock the clients map to safely access client nodes
        let clients = self.clients.lock().unwrap();

        // Count clients that share the same virt_server by using Arc::ptr_eq
        clients.values().filter(|client| {
            if let Some(client_virt_server) = &client.virt_server {
                Arc::ptr_eq(client_virt_server, &virt_server)
            } else {
                false
            }
        }).count()
    }

    pub fn add_client(&self, mut client: FlytClientNode) {
        let mut clients = self.clients.lock().unwrap();
        let grouping_enabled = {
            let ip_grouping = self.ip_grouping_enabled.lock().unwrap();
            *ip_grouping.get(&client.ipaddr).unwrap_or(&false)
        };

        let group_id = if grouping_enabled {
            let mut ip_to_group = self.ipaddr_to_group_id.lock().unwrap();
            *ip_to_group.entry(client.ipaddr.clone()).or_insert_with(|| {
                Uuid::new_v4().as_u128() as i32
            })
        } else {
            -1
        };

        if group_id != -1 {
            // Lock the group_to_virt_server map to safely update it
            let mut group_to_virt = self.group_to_virt_server.lock().unwrap();

            // Update or insert the virt_server for the given group_id
            group_to_virt.insert(group_id, client.virt_server.clone().expect("virt_server not present"));
        }

        client.group_id = group_id;
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

    pub fn get_clients_by_group(&self, ipaddr: &str, client_id: i32) -> Vec<(String, i32)> {
        let clients = self.clients.lock().unwrap();

        // Create a vector to hold the results
        let mut result = Vec::new();

        // Check if the client exists
        if let Some(client) = clients.get(&(ipaddr.to_string(), client_id)) {
            let group_id = client.group_id;

            // If the client has a valid group_id (not -1)
            if group_id != -1 {
                // Collect all clients with the same group_id
                for (key, client) in clients.iter() {
                    if client.group_id == group_id {
                        result.push((key.0.clone(), key.1));  // Add (ipaddr, client_id)
                    }
                }
            } else {
                // If the client has no group (group_id == -1), return just this client
                result.push((ipaddr.to_string(), client_id));
            }
        }

        result
    }

    pub fn stop_client(&self, ipaddr: &str, client_id: i32) -> Result<(),String> {
       let result = self.get_clients_by_group(ipaddr, client_id);
       let mut errors = Vec::new(); // Collect errors in this vector

       for (ipaddr, client_id) in result.iter() {
        let client_id = *client_id;
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
                                errors.push(format!("Error reading response from client {}: {}", ipaddr, e));
                                continue; // Continue to the next client
                            }
                        };
                        if response[0] != "200" {
                            errors.push(format!("Error stopping client: {}. Response: {}", client_id, response[1]));
                            continue; // Continue to the next client
                        }
                    }
                    Err(e) => {
                        log::error!("Error writing to client: {}", e);
                        errors.push(format!("Error writing to client {}: {}", ipaddr, e));
                        continue; // Continue to the next client
                    }
                }
            }
            else {
                log::info!("Client stream is None");
                errors.push(format!("Client stream is None for client: {},{}", ipaddr, client_id));
                continue; // Continue to the next client
            }
        }
        else {
            errors.push(format!("Client not found: {},{}", ipaddr, client_id));
            continue; // Continue to the next client
        }
      }
      if errors.is_empty() {
        Ok(())
      } else {
        Err(errors.join("\n")) // Join all errors into a single string and return as Err
      }
    }

    pub fn change_virt_server(&self, ipaddr: &str, client_id: i32, new_virt_server: &Arc<RwLock<VirtServer>>) -> Result<(), String> {
       let result = self.get_clients_by_group(ipaddr, client_id);
       let mut errors = Vec::new(); // Collect errors in this vector
       for (ipaddr, client_id) in result.iter() {
        let client_id = *client_id;
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
                                errors.push(format!("Error reading response from client {}: {}", ipaddr, e));
                                continue; // Continue to the next client
                            }
                        };
                        if response[0] != "200" {
                            errors.push(format!("Error stopping client: {}. Response: {}", client_id, response[1]));
                            continue; // Continue to the next client
                        }
                        
                        // Update client virt server
                        client.virt_server = Some(new_virt_server.clone());
                        self.update_client(client);
                    }
                    Err(e) => {
                        log::error!("Error writing to client: {}", e);
                        errors.push(format!("Error writing to client {}: {}", ipaddr, e));
                        continue; // Continue to the next clien
                    }
                }
            }
            else {
                log::info!("Client stream is None");
                errors.push(format!("Client stream is None for client: {},{}", ipaddr, client_id));
                continue; // Continue to the next client
            }
        }
        else {
            errors.push(format!("Client not found: {},{}", ipaddr, client_id));
            continue; // Continue to the next clien
        }
      }
      if errors.is_empty() {
        Ok(())
      } else {
        Err(errors.join("\n")) // Join all errors into a single string and return as Err
      }
    }

    pub fn resume_client(&self, ipaddr: &str, client_id: i32) -> Result<(),String> {
       let result = self.get_clients_by_group(ipaddr, client_id);
       let mut errors = Vec::new(); // Collect errors in this vector
       for (ipaddr, client_id) in result.iter() {
        let client_id = *client_id;
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
                                errors.push(format!("Error reading response from client {}: {}", ipaddr, e));
                                continue; // Continue to the next client
                            }
                        };
                        if response[0] != "200" {
                            errors.push(format!("Error stopping client: {}. Response: {}", client_id, response[1]));
                            continue; // Continue to the next client
                        }
                    }
                    Err(e) => {
                        log::error!("Error writing to client: {}", e);
                        errors.push(format!("Error writing to client {}: {}", ipaddr, e));
                        continue; // Continue to the next client
                    }
                }
            }
            else {
                log::info!("Client stream is None");
                errors.push(format!("Client stream is None for client: {},{}", ipaddr, client_id));
                continue; // Continue to the next client
            }
        }
        else {
            log::info!("Client not found: {},{}", ipaddr, client_id);
            errors.push(format!("Client not found: {},{}", ipaddr, client_id));
            continue; // Continue to the next client
        }
      }
      if errors.is_empty() {
        Ok(())
      } else {
        Err(errors.join("\n")) // Join all errors into a single string and return as Err
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

    fn get_client_command_client_id(response_str: String) -> (String, i32, i32) {
        let client_details = response_str.split(",").collect::<Vec<&str>>();
        if client_details.len() < 2 || client_details.len() > 3 {
            log::error!("client details not in correct format: {}", response_str);
            return ("".to_string(), -1, -1);
        }

        let first_value = client_details[0].to_string();
        let second_value = client_details[1].trim().parse::<i32>().unwrap_or_else(|_| {
            log::error!("Failed to parse second value as i32: {}", client_details[1]);
            -1
        });
        let third_value = if client_details.len() == 3 {
            client_details[2].trim().parse::<i32>().unwrap_or_else(|_| {
                log::error!("Failed to parse third value as i32: {}", client_details[2]);
                -1
            })
        } else {
            -1
        };

        (first_value, second_value, third_value)
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
        
        let (command, client_id, sm_core) = match StreamUtils::read_response(&mut reader, 1) {
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
                    log::info!("creating new client {},{}, sm_core: {}", client_ip, client_id, sm_core);
                    // Try to get an existing virt_server for the IP, or allocate a new one if it doesn't exist.
                    let virt_server = match self.get_virt_server_for_ip(&client_ip) {
                        Some(vs) => vs, // Use the existing virt_server if found.
                        None => {
                            // Try to allocate VM resources if no existing virt_server is found.
                            let old_compute_units = self.server_nodes_manager.vm_resource_getter.set_vm_sm_resource(&client_ip,sm_core);
                            match self.server_nodes_manager.allocate_vm_resources(&client_ip, client_id) {
                                Ok(vs) => {
                                    if !old_compute_units.is_none() {
                                        self.server_nodes_manager.vm_resource_getter.set_vm_sm_resource(&client_ip, old_compute_units.expect("not valid old compute"));
                                    }
                                    vs // Use the newly allocated virt_server.
                                }
                                Err(e) => {
                                    log::error!("Failed to allocate VM resources for {},{}: {}", client_ip, client_id, e);
                                    if !old_compute_units.is_none() {
                                        self.server_nodes_manager.vm_resource_getter.set_vm_sm_resource(&client_ip, old_compute_units.expect("not valid old compute"));
                                    }
                                    let _ = stream.write_all(format!("500\nFailed to allocate VM resources error: {}\n",e).as_bytes());
                                    return;
                                }
                            }
                        }
                    };
                    let mut vmr = self.server_nodes_manager.vm_resource_getter.get_vm_required_resources(&client_ip).unwrap();
                    if sm_core > 0 {
                        vmr.compute_units = sm_core as u32;
                    }

                    log::info!("after allocating vm resources new client {},{}", client_ip, client_id);
                    let client = FlytClientNode {
                            ipaddr: client_ip.clone(),
                            client_id: client_id,
                            group_id: -1,
                            stream: Arc::new(RwLock::new(Some(StreamEnds{writer: stream, reader}))),
                            virt_server: Some(virt_server.clone()),
                            is_migrating: RwLock::new(false),
                            is_active: RwLock::new(true),
                            compute_requested: vmr.compute_units,
                            memory_requested: vmr.memory,
                            server_metrics: Vec::new(),
                            client_metrics: Vec::new(),
                    };
                    let client_stream_clone = client.stream.clone();
                    let mut client_stream_clone_writer = client_stream_clone.write().unwrap();
                    let virt_server = virt_server.read().unwrap();
                    match client_stream_clone_writer.as_mut().unwrap().writer.write_all(format!("200\n{},{}\n", virt_server.ipaddr, virt_server.rpc_id).as_bytes()) {
                        Ok(_) => {
                            log::info!("Adding new client {},{}", client_ip, client_id);
                            self.add_client(client);
                        }
                        Err(e) => {
                            log::error!("Error writing to stream: {}", e);
                        }
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

                        if self.count_clients_with_virt_server(client.virt_server.as_ref().unwrap().clone()) == 1 {
                            self.server_nodes_manager.free_virt_server(&virt_server_ip, rpc_id, false);
                        }
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

