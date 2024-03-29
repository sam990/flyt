
use crate::bookkeeping::*;
use crate::servernode_handler::ServerNodesManager;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;


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

    // pub fn update_client(&self, client: FlytClientNode) {
    //     self.add_client(client);
    // }

    // pub fn get_client(&self, ipaddr: &str) -> Option<FlytClientNode> {
    //     let clients = self.clients.lock().unwrap();
    //     clients.get(ipaddr).cloned()
    // }

    // pub fn remove_client(&self, ipaddr: &str) {
    //     let mut clients = self.clients.lock().unwrap();
    //     clients.remove(ipaddr);
    // }

    // pub fn get_all_clients(&self) -> Vec<FlytClientNode> {
    //     let clients = self.clients.lock().unwrap();
    //     clients.values().cloned().collect()
    // }

    pub fn exists(&self, ipaddr: &str) -> bool {
        let clients = self.clients.lock().unwrap();
        clients.contains_key(ipaddr)
    }


    pub fn start_flytclient_handler(&self, port: u16) {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    self.handle_flytclient(stream)
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
        }
    }

    fn handle_flytclient(&self, mut stream: TcpStream) {
        let client_ip = stream.peer_addr().unwrap().ip().to_string();
        
        if self.exists(&client_ip) {
            println!("Client already exists: {}", client_ip);
            // handle_client_command
            return;
        }

        println!("New client connected: {}", client_ip);
        // send client details
        let allocated_virt_server = self.server_nodes_manager.allocate_vm_resources(&client_ip);

        match allocated_virt_server {
            Ok(virt_server) => {
                
                stream.write_all(format!("200\n{},{}\n", virt_server.ipaddr, virt_server.rpc_id).as_bytes()).unwrap();

                let client = FlytClientNode {
                    ipaddr: client_ip.clone(),
                    stream: stream,
                    virt_server: virt_server,
                };
                self.add_client(client);
            }
            Err(error_string) => {
                println!("No server available for client: {}\nError: {}", client_ip, error_string);
                // send no server available message
            }
        }

        
    }

    
}

