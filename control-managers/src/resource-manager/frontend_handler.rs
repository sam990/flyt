use std::{io::{BufRead, BufReader, Write}, os::unix::net::{UnixListener, UnixStream}};

use crate::{client_handler::FlytClientManager, common::api_commands::FrontEndCommand, servernode_handler::ServerNodesManager};




pub struct InputHandler<'a> {
    client_mgr: &'a FlytClientManager<'a>,
    server_nodes_manager: &'a ServerNodesManager<'a>,
}

impl <'a> InputHandler<'a> {
    pub fn new(client_mgr: &'a FlytClientManager, server_nodes_manager: &'a ServerNodesManager) -> Self {
        InputHandler {
            client_mgr,
            server_nodes_manager,
        }
    }

    pub fn start_listening(&self, socket_path: &str) {
        let listener = UnixListener::bind(socket_path).unwrap();
        for mut stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    self.handle_request(stream);
                }
                Err(e) => {
                    println!("Error: {}", e);
                    break;
                }
            }
        }
    }

    fn handle_request(&self, stream: UnixStream) {
        let reader_clone = stream.try_clone().unwrap();
        let mut reader = BufReader::new(reader_clone);
        let mut buffer = String::new();
        reader.read_line(&mut buffer).unwrap();
        
        match buffer.trim() {
            FrontEndCommand::LIST_VMS => {
                self.list_vms(stream);
            }
            FrontEndCommand::LIST_SERVER_NODES => {
                self.list_servernodes(stream);
            }
            _ => {
                println!("Invalid command");
            }
        }
    }

    fn list_vms(&self, mut stream: UnixStream){
        let vms = self.client_mgr.get_all_clients();
        let mut response = String::new();
        for vm in vms {
            // format: vmip,servnode_ip,servnode_rpcid,sm_cores,memory,isactive
            response.push_str(&format!("{},{},{},{},{},{}\n",
                vm.ipaddr,
                if vm.virt_server.is_some() { vm.virt_server.as_ref().unwrap().read().unwrap().ipaddr.clone() } else { String::new() },
                if vm.virt_server.is_some() { vm.virt_server.as_ref().unwrap().read().unwrap().rpc_id.to_string() } else { String::new() },
                if vm.virt_server.is_some() { vm.virt_server.as_ref().unwrap().read().unwrap().compute_units.to_string() } else { String::new() },
                if vm.virt_server.is_some() { vm.virt_server.as_ref().unwrap().read().unwrap().memory.to_string() } else { String::new() },
                vm.is_active.read().unwrap().to_string()

            ));
        }
        stream.write_all(response.as_bytes()).unwrap();
    }

    fn list_servernodes(&self, mut stream: UnixStream) {
        let server_nodes = self.server_nodes_manager.get_all_server_nodes();
        let mut response = String::new();
        for server_node in server_nodes {
            // format: ipaddr,num_vgpus
            response.push_str(&format!("{},{}\n", server_node.ipaddr, server_node.gpus.len()));

            for gpu in server_node.gpus.iter() {
                // format: gpuid,name,memory,allocated_memory,compute_units,allocated_compute_units
                response.push_str(&format!("{},{},{},{},{},{}\n",
                    gpu.read().unwrap().gpu_id,
                    gpu.read().unwrap().name,
                    gpu.read().unwrap().memory,
                    gpu.read().unwrap().allocated_memory,
                    gpu.read().unwrap().compute_units,
                    gpu.read().unwrap().allocated_compute_units
                ));
            }
        }
        stream.write_all(response.as_bytes()).unwrap();
    }

    fn list_virt_servers(&self, mut stream: UnixStream) {
        let mut response = String::new();
        let serv_nodes = self.server_nodes_manager.get_all_server_nodes();
        for serv_node in serv_nodes {
            for virt_server in serv_node.virt_servers.iter() {
                // format: ipaddr,rpc_id,gpu_id,compute_units,memory
                let virt_server = virt_server.read().unwrap();
                response.push_str(&format!("{},{},{},{},{}\n",
                    virt_server.ipaddr,
                    virt_server.rpc_id,
                    virt_server.gpu.read().unwrap().gpu_id,
                    virt_server.compute_units,
                    virt_server.memory,
                ));
            }
        }
        stream.write_all(response.as_bytes()).unwrap();
    }
    
}