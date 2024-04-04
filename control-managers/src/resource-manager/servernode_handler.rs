

use crate::bookkeeping::*;
use crate::common::api_commands::FlytApiCommand;
use crate::common::utils::Utils;

use std::collections::HashMap;
use std::io::{ BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};

pub struct ServerNodesManager<'a> {
    server_nodes: Mutex<HashMap<String, ServerNode>>,
    vm_resource_getter: &'a VMResourcesGetter,
}

impl<'a> ServerNodesManager<'a> {

    pub fn new( resource_getter: &'a VMResourcesGetter ) -> Self {
        ServerNodesManager {
            server_nodes: Mutex::new(HashMap::new()),
            vm_resource_getter: resource_getter,
        }
    }

    pub fn add_server_node(&self, server_node: ServerNode) {
        let mut server_nodes = self.server_nodes.lock().unwrap();
        server_nodes.insert(server_node.ipaddr.clone(), server_node);
    }

    pub fn update_server_node(&self, server_node: ServerNode) {
        self.add_server_node(server_node);
    }

    pub fn get_server_node(&self, ipaddr: &String) -> Option<ServerNode> {
        let server_nodes = self.server_nodes.lock().unwrap();
        server_nodes.get(ipaddr).cloned()
    }

    pub fn remove_server_node(&self, ipaddr: &str) {
        let mut server_nodes = self.server_nodes.lock().unwrap();
        server_nodes.remove(ipaddr);
    }

    pub fn get_all_server_nodes(&self) -> Vec<ServerNode> {
        let server_nodes = self.server_nodes.lock().unwrap();
        server_nodes.values().cloned().collect()
    }

    pub fn exists(&self, ipaddr: &str) -> bool {
        let server_nodes = self.server_nodes.lock().unwrap();
        server_nodes.contains_key(ipaddr)
    }
    
    pub fn start_servernode_handler(&self, port : u16) {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    self.handle_servernode(stream)
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
        }
    }

    fn handle_servernode(&self, stream: TcpStream) {
        let server_ip = stream.peer_addr().unwrap().ip().to_string();

        let server_node = ServerNode {
            ipaddr: server_ip.clone(),
            gpus: Vec::new(),
            stream: stream.try_clone().unwrap(),
            virt_servers: Vec::new(),
        };
    
        if self.exists(&server_node.ipaddr) {
            println!("Server node already exists: {}", server_node.ipaddr);
            return;
        }
        
        self.add_server_node(server_node);
        let _ = self.update_server_node_gpus(&server_ip);
        
    }

    fn update_server_node_gpus(&self, server_node_ip: &String ) -> Result<(),String> {

        if !self.exists(server_node_ip) {
            println!("Server node not found: {}", server_node_ip);
            return Err("Server node not found".to_string());
        }

        let mut server_node = self.get_server_node(server_node_ip).unwrap();

        let stream_clone = server_node.stream.try_clone().unwrap();
        let mut reader = BufReader::new(stream_clone);
        let mut buffer = String::new();

        server_node.stream.write_all(format!("{}\n", FlytApiCommand::RMGR_SNODE_SEND_GPU_INFO).as_bytes()).unwrap();

        reader.read_line(&mut buffer).unwrap();

        if buffer.trim() != "200" {
            let status_code = buffer.clone();
            reader.read_line(&mut buffer).unwrap();
            println!("RMGR_SNODE_SEND_GPU_INFO, Status: {}\n{}", status_code, buffer);
            return Err(format!("RMGR_SNODE_SEND_GPU_INFO, Status: {}\n{}", status_code, buffer));
        }

        reader.read_line(&mut buffer).unwrap();
        let num_gpus = buffer.trim().parse::<u64>().unwrap();

        let mut gpus = Vec::new();

        for _ in 0..num_gpus {
            reader.read_line(&mut buffer).unwrap();
            let gpu_info = buffer.trim().split(",").collect::<Vec<&str>>();
            // name, memory, compute_units, compute_power, gpu_id
            let gpu = Arc::new(RwLock::new (GPU {
                name: gpu_info[0].to_string(),
                memory: gpu_info[1].parse::<u64>().unwrap(),
                compute_units: gpu_info[2].parse::<u64>().unwrap(),
                compute_power: gpu_info[3].parse::<u64>().unwrap(),
                gpu_id: gpu_info[4].parse::<u64>().unwrap(),
                ..Default::default()
            }));
            gpus.push(gpu);
        }

        server_node.gpus = gpus;
        self.update_server_node(server_node);
        Ok(())
    }

    pub fn allocate_vm_resources(&self, client_ip: &String,) -> Result<VirtServer,String> {
        let vm_required_resources = self.vm_resource_getter.get_vm_required_resources(client_ip);
        
        if vm_required_resources.is_none() {
            println!("VM resources not found for client: {}", client_ip);
            return Err("VM resources not found".to_string());
        }

        let vm_required_resources = vm_required_resources.unwrap();

        // First checking if the server on which client is running has enough resources
        let host_server_node = self.get_server_node(&vm_required_resources.host_ip);
        let mut target_server_ip : Option<String> = None;
        let mut target_gpu_id : Option<u64> = None;
        
        if host_server_node.is_some() {
            let host_server_node = host_server_node.unwrap();
            let gpu_id = check_resource_availability(&host_server_node, &vm_required_resources);
            if gpu_id.is_some() {
                target_server_ip = Some(host_server_node.ipaddr);
                target_gpu_id = gpu_id;
            }
        }

        // If not, then checking all other servers
        if target_server_ip.is_none() {
            let server_nodes = self.get_all_server_nodes();
            for server_node in server_nodes {
                let gpu_id = check_resource_availability(&server_node, &vm_required_resources);
                if gpu_id.is_some() {
                    target_server_ip = Some(server_node.ipaddr);
                    target_gpu_id = gpu_id;
                    break;
                }
            }
        }

        if target_server_ip.is_none() {
            println!("No server found with enough resources for client: {}", client_ip);
            return Err("No server found with enough resources".to_string());
        }


        // communicate with the node
        let mut target_server_node = self.get_server_node(&target_server_ip.unwrap()).unwrap();
        let stream_clone = target_server_node.stream.try_clone().unwrap();
        
        let mut reader = BufReader::new(stream_clone);
        let mut status = String::new();
        let mut payload = String::new();

        target_server_node.stream.write_all(format!("{}\n{},{},{}\n", FlytApiCommand::RMGR_SNODE_ALLOC_VIRT_SERVER, target_gpu_id.unwrap(), vm_required_resources.compute_units, vm_required_resources.memory).as_bytes()).unwrap();

        reader.read_line(&mut status).unwrap();
        reader.read_line(&mut payload).unwrap();

        if status.trim() != "200" {
            println!("RMGR_SNODE_ALLOC_VIRT_SERVER, Status: {}\n{}", status, payload);
            return Err(format!("RMGR_SNODE_ALLOC_VIRT_SERVER, Status: {}\n{}", status, payload));
        }
        
        let target_gpu_id = target_gpu_id.unwrap();
        let virt_server_rpc_id = payload.trim().parse::<u64>().unwrap();

        let target_gpu = target_server_node.gpus.iter_mut().find(|gpu| gpu.read().unwrap().gpu_id == target_gpu_id).unwrap();
        
        
        // rwlock guard block
        {
            let mut gpu_write = target_gpu.write().unwrap();
            gpu_write.allocated_compute_units += vm_required_resources.compute_units;
            gpu_write.allocated_memory += vm_required_resources.memory;
        }

        
        let virt_server =VirtServer {
            ipaddr: client_ip.clone(),
            compute_units: vm_required_resources.compute_units,
            memory: vm_required_resources.memory,
            rpc_id: virt_server_rpc_id as u16,
            gpu: target_gpu.clone(),
        };

        target_server_node.virt_servers.push(virt_server.clone());
        
        self.update_server_node(target_server_node);

        Ok(virt_server)

    }

    pub fn free_virt_server(&self, virt_ip: String, rpc_id: u16) -> Result<(),String> {
        
        let server_node = self.get_server_node(&virt_ip);

        if server_node.is_none() {
            println!("Server node not found: {}", virt_ip);
            return Err("Server node not found".to_string());
        }

        let mut server_node = server_node.unwrap();

        let target_vserver = server_node.virt_servers.iter().find(|virt_server| virt_server.rpc_id == rpc_id);

        if target_vserver.is_none() {
            println!("Virt server not found: {}", rpc_id);
            return Err("Virt server not found".to_string());
        }


        server_node.stream.write_all(format!("{}\n{}\n", FlytApiCommand::RMGR_SNODE_DEALLOC_VIRT_SERVER, rpc_id).as_bytes()).unwrap();
        let response = Utils::read_response(&mut server_node.stream, 2);

        if response[0] != "200" {
            println!("RMGR_SNODE_DEALLOC_VIRT_SERVER, Status: {}\n{}", response[0], response[1]);
            return Err(format!("RMGR_SNODE_DEALLOC_VIRT_SERVER, Status: {}\n{}", response[0], response[1]));
        }

        let target_vserver = target_vserver.unwrap();
        
        target_vserver.gpu.write().unwrap().allocated_compute_units -= target_vserver.compute_units;
        target_vserver.gpu.write().unwrap().allocated_memory -= target_vserver.memory;

        server_node.virt_servers.retain(|virt_server| virt_server.rpc_id != rpc_id);
        
        Ok(())
    }

    

}


fn check_resource_availability(server_node: &ServerNode, vm_resources: &VMResources) -> Option<u64> {
    for gpu in server_node.gpus.iter() {
        let gpu_read = gpu.read().unwrap();
        let remain_compute_units = gpu_read.compute_units - gpu_read.allocated_compute_units;
        let remain_memory = gpu_read.memory - gpu_read.allocated_memory;
        if remain_memory >= vm_resources.memory && remain_compute_units >= vm_resources.compute_units {
            return Some(gpu_read.gpu_id);
        }
    }
    None
}