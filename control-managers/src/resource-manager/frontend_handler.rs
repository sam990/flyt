use std::{io::{BufReader, Write}, os::unix::net::{UnixListener, UnixStream}};

use crate::{client_handler::FlytClientManager, common::{api_commands::FrontEndCommand, utils::Utils}, servernode_handler::ServerNodesManager};


#[derive(PartialEq)]
enum ChangeConfigFor {
    SmCores,
    Memory,
    Both,
}

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
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
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
        
        let command = Utils::read_line(&mut reader);
        
        match command.as_str() {
            FrontEndCommand::LIST_VMS => {
                self.list_vms(stream);
            }
            FrontEndCommand::LIST_SERVER_NODES => {
                self.list_servernodes(stream);
            }
            FrontEndCommand::LIST_VIRT_SERVERS => {
                self.list_virt_servers(stream);
            }
            FrontEndCommand::CHANGE_SM_CORES => {
                self.change_resources(stream, ChangeConfigFor::SmCores);
            }
            FrontEndCommand::CHANGE_MEMORY => {
                self.change_resources(stream, ChangeConfigFor::Memory);
            }
            FrontEndCommand::CHANGE_SM_CORES_AND_MEMORY => {
                self.change_resources(stream, ChangeConfigFor::Both);
            }
            _ => {
                println!("Invalid command");
            }
        }
    }

    fn list_vms(&self, mut stream: UnixStream){
        let vms = self.client_mgr.get_all_clients();
        let mut response = String::new();
        response.push_str(format!("200\n{}\n", vms.len()).as_str());
        for vm in vms {
            // format: vmip,servnode_ip,servnode_rpcid,sm_cores,memory,isactive
            if let Some(virt_server) = vm.virt_server {
                let virt_server = virt_server.read().unwrap();
                response.push_str(&format!("{},{},{},{},{},{}\n",
                    vm.ipaddr,
                    virt_server.ipaddr,
                    virt_server.rpc_id,
                    virt_server.compute_units,
                    virt_server.memory,
                    *vm.is_active.read().unwrap()
                ));
            }
            else {
                response.push_str(&format!("{},,,,{},\n",
                    vm.ipaddr,
                    *vm.is_active.read().unwrap()
                ));
            }
            
        }
        stream.write_all(response.as_bytes()).unwrap();
    }

    fn list_servernodes(&self, mut stream: UnixStream) {
        let server_nodes = self.server_nodes_manager.get_all_server_nodes();
        let mut response = String::new();
        response.push_str(format!("200\n{}\n", server_nodes.len()).as_str());
        for server_node in server_nodes {
            // format: ipaddr,num_vgpus
            response.push_str(&format!("{},{}\n", server_node.ipaddr, server_node.gpus.len()));

            for gpu in server_node.gpus.iter() {
                // format: gpuid,name,memory,allocated_memory,compute_units,allocated_compute_units
                let gpu = gpu.read().unwrap();
                response.push_str(&format!("{},{},{},{},{},{}\n",
                    gpu.gpu_id,
                    gpu.name,
                    gpu.memory,
                    gpu.allocated_memory,
                    gpu.compute_units,
                    gpu.allocated_compute_units
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
        response.insert_str(0, format!("200\n{}\n", response.lines().count()).as_str());
        stream.write_all(response.as_bytes()).unwrap();
    }

    fn change_resources(&self, mut stream: UnixStream, change_for: ChangeConfigFor) {
        let buffer = Utils::read_response(stream.try_clone().unwrap().by_ref(), 1);
        let parts: Vec<&str> = buffer[0].split(',').collect();
        if (change_for != ChangeConfigFor::Both && parts.len() != 2) || (change_for == ChangeConfigFor::Both && parts.len() != 3 ){
            stream.write_all(b"Invalid command").unwrap();
            return;
        }
        let ipaddr = parts[0];
        let new_resource = parts[1].parse::<u64>().unwrap();

        let client = self.client_mgr.get_client(ipaddr);
        if client.is_none() {
            stream.write_all(b"500\nVM not found\n").unwrap();
            return;
        }

        let client = client.unwrap();
        
        if client.virt_server.is_none() {
            stream.write_all(b"500\nVM is not running on any server node\n").unwrap();
            return;
        }

        let (virt_server_ip, virt_server_rpc_id, cur_compute, cur_mem) = {
            let virt_server = client.virt_server.as_ref().unwrap().read().unwrap();
            (virt_server.ipaddr.clone(), virt_server.rpc_id, virt_server.compute_units, virt_server.memory)
        };

        let ret = match change_for {
            ChangeConfigFor::SmCores => self.server_nodes_manager.change_resource_configurations(&virt_server_ip, virt_server_rpc_id, new_resource as u32, cur_mem),
            ChangeConfigFor::Memory => self.server_nodes_manager.change_resource_configurations(&virt_server_ip, virt_server_rpc_id, cur_compute, new_resource),
            ChangeConfigFor::Both => {
                let mem_new = parts[2].parse::<u64>().unwrap();
                self.server_nodes_manager.change_resource_configurations(&virt_server_ip, virt_server_rpc_id, new_resource as u32, mem_new)
            }
        };

        if ret.is_ok() {
            stream.write_all(b"200\nResource updated successfully\n").unwrap();
        }
        else {
            stream.write_all(b"500\nFailed to update resource\n").unwrap();
        }

    }
    
}