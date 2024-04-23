

use log::info;

use crate::bookkeeping::*;
use crate::common::api_commands::FlytApiCommand;
use crate::common::types::StreamEnds;
use crate::common::utils::StreamUtils;

use std::collections::HashMap;
use std::io::{ BufReader, Write };
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};


macro_rules! get_reader {
    ($server_node: expr) => {
        $server_node.stream.write().unwrap().reader
    };
}

macro_rules! get_writer {
    ($server_node: expr) => {
        $server_node.stream.write().unwrap().writer
    };
}


macro_rules! stream_write {
    ($stream: expr, $data: expr) => {
        match $stream.write_all($data.as_bytes()) {
            Ok(_) => {}
            Err(e) => {
                log::error!("Error writing to stream: {}", e);
                return Err(format!("Error writing to stream: {}", e));
            }
        }
    };
}

macro_rules! stream_read_line {
    ($stream: expr) => {
        match StreamUtils::read_line(&mut $stream) {
            Ok(data) => data,
            Err(e) => {
                log::error!("Error reading from stream: {}", e);
                return Err(format!("Error reading from stream: {}", e));
            }
        }
    };
}

macro_rules! stream_read_response {
    ($reader: expr, $num_lines: expr) => {
        match StreamUtils::read_response(&mut $reader, $num_lines) {
            Ok(response) => response,
            Err(e) => {
                log::error!("Error reading response from stream: {}", e);
                return Err(format!("Error reading response from stream: {}", e));
            }
        }
    };
}

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
        log::info!("Server node handler started on port: {}", port);
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    self.handle_servernode(stream)
                }
                Err(e) => {
                    log::error!("Error accepting connection: {}", e)
                }
            }
        }
    }

    fn handle_servernode(&self, stream: TcpStream) {
        let server_ip = match stream.peer_addr() {
            Ok(addr) => addr.ip().to_string(),
            Err(e) => {
                log::error!("Error getting ip address of serverndoe: {}", e);
                return;
            }
        };

        let reader_clone = match stream.try_clone() {
            Ok(stream) => stream,
            Err(e) => {
                log::error!("Error cloning stream: {}", e);
                return;
            }
        };

        let reader = BufReader::new(reader_clone);
    
        log::info!("Server node connected: {}", server_ip);

        let server_node = ServerNode {
            ipaddr: server_ip.clone(),
            gpus: Vec::new(),
            stream: Arc::new(RwLock::new(StreamEnds{writer: stream, reader })),
            virt_servers: Vec::new(),
        };
    
        if self.exists(&server_node.ipaddr) {
            log::error!("Server node already exists: {}", server_node.ipaddr);
            return;
        }
        
        self.add_server_node(server_node);
        let _ = self.update_server_node_gpus(&server_ip);
        
    }

    fn update_server_node_gpus(&self, server_node_ip: &String ) -> Result<(),String> {

        log::info!("Getting GPU details for servernode: {}", server_node_ip);

        if !self.exists(server_node_ip) {
            log::error!("Server node not found: {}", server_node_ip);
            return Err("Server node not found".to_string());
        }

        let mut server_node = self.get_server_node(server_node_ip).unwrap();

        stream_write!(get_writer!(server_node), format!("{}\n", FlytApiCommand::RMGR_SNODE_SEND_GPU_INFO));

        let status = stream_read_line!(get_reader!(server_node));

        if status != "200" {
            let err_msg = stream_read_line!(get_reader!(server_node));
            log::error!("RMGR_SNODE_SEND_GPU_INFO, Status: {}\n{}", status, err_msg);
            return Err(format!("RMGR_SNODE_SEND_GPU_INFO, Status: {}\n{}", status, err_msg));
        }
        
        let num_gpus_str = stream_read_line!(get_reader!(server_node));
        
        let num_gpus = num_gpus_str.parse::<u64>().unwrap();

        let mut gpus = Vec::new();

        for _ in 0..num_gpus {
            let gpu_info_str = stream_read_line!(get_reader!(server_node));
            let gpu_info = gpu_info_str.split(",").collect::<Vec<&str>>();
            // gpu_id, name, memory, compute_units, compute_power
            let gpu = Arc::new(RwLock::new (GPU {
                gpu_id: gpu_info[0].parse::<u64>().unwrap(),
                name: gpu_info[1].to_string(),
                memory: gpu_info[2].parse::<u64>().unwrap(),
                compute_units: gpu_info[3].parse::<u32>().unwrap(),
                compute_power: gpu_info[4].parse::<u64>().unwrap(),
                ..Default::default()
            }));
            gpus.push(gpu);
        }

        server_node.gpus = gpus;
        self.update_server_node(server_node);
        log::info!("Server node gpus updated: {}", server_node_ip);
        Ok(())
    }

    pub fn allocate_vm_resources(&self, client_ip: &String,) -> Result<Arc<RwLock<VirtServer>>,String> {
        let vm_required_resources = self.vm_resource_getter.get_vm_required_resources(client_ip);
        
        log::info!("Allocating VM resources for client: {}", client_ip);
        log::info!("VM resources required: {:?}", vm_required_resources);

        if vm_required_resources.is_none() {
            log::error!("VM resources not found for client: {}", client_ip);
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
            log::error!("No server found with enough resources for client: {}", client_ip);
            return Err("No server found with enough resources".to_string());
        }

        let target_server_ip = target_server_ip.unwrap();


        // communicate with the node
        let mut target_server_node = self.get_server_node(&target_server_ip).unwrap();

        stream_write!(get_writer!(target_server_node), format!("{}\n{},{},{}\n", FlytApiCommand::RMGR_SNODE_ALLOC_VIRT_SERVER, target_gpu_id.unwrap(), vm_required_resources.compute_units, vm_required_resources.memory));

        let status = stream_read_line!(get_reader!(target_server_node));
        let payload = stream_read_line!(get_reader!(target_server_node));

        if status != "200" {
            log::error!("RMGR_SNODE_ALLOC_VIRT_SERVER, Status: {}: {}", status, payload);
            return Err(format!("RMGR_SNODE_ALLOC_VIRT_SERVER, Status: {}: {}", status, payload));
        }
        
        let target_gpu_id = target_gpu_id.unwrap();
        let virt_server_rpc_id = payload.parse::<u64>().unwrap();

        let target_gpu = target_server_node.gpus.iter_mut().find(|gpu| gpu.read().unwrap().gpu_id == target_gpu_id).unwrap();
        
        
        // rwlock guard block
        {
            let mut gpu_write = target_gpu.write().unwrap();
            gpu_write.allocated_compute_units += vm_required_resources.compute_units;
            gpu_write.allocated_memory += vm_required_resources.memory;
        }

        log::info!("Virt server {}/{} allocated for client: {}", target_server_ip, virt_server_rpc_id, client_ip);
        
        let virt_server = Arc::new(RwLock::new(VirtServer {
            ipaddr: target_server_ip,
            compute_units: vm_required_resources.compute_units,
            memory: vm_required_resources.memory,
            rpc_id: virt_server_rpc_id as u64,
            gpu: target_gpu.clone(),
        }));

        target_server_node.virt_servers.push(virt_server.clone());
        
        self.update_server_node(target_server_node);


        Ok(virt_server)

    }

    pub fn free_virt_server(&self, virt_ip: String, rpc_id: u64) -> Result<(),String> {

        info!("Deallocating virt server: {}/{}", virt_ip, rpc_id);

        let server_node = self.get_server_node(&virt_ip);

        if server_node.is_none() {
            log::error!("Server node not found: {}", virt_ip);
            return Err("Server node not found".to_string());
        }

        let mut server_node = server_node.unwrap();

        let target_vserver = server_node.virt_servers.iter().find(|virt_server| virt_server.read().unwrap().rpc_id == rpc_id);

        if target_vserver.is_none() {
            log::error!("Virt server not found: {}", rpc_id);
            return Err("Virt server not found".to_string());
        }

        log::trace!("Sending dealloc command to server node: {}/{}", virt_ip, rpc_id);
        stream_write!(get_writer!(server_node), format!("{}\n{}\n", FlytApiCommand::RMGR_SNODE_DEALLOC_VIRT_SERVER, rpc_id));
        
        let response = stream_read_response!(get_reader!(server_node), 2);

        log::trace!("Response from server node {} for deallocate: {:?}", virt_ip, response);

        if response[0] != "200" {
            log::error!("RMGR_SNODE_DEALLOC_VIRT_SERVER, Status: {}\n{}", response[0], response[1]);
            return Err(format!("RMGR_SNODE_DEALLOC_VIRT_SERVER, Status: {}\n{}", response[0], response[1]));
        }

        let target_vserver = target_vserver.unwrap();

        {
            let target_vserver_lock_guard = target_vserver.read().unwrap();
            let mut gpu_write_lock_guard = target_vserver_lock_guard.gpu.write().unwrap();

            gpu_write_lock_guard.allocated_compute_units -= target_vserver_lock_guard.compute_units;
            gpu_write_lock_guard.allocated_memory -= target_vserver_lock_guard.memory;
        }

        server_node.virt_servers.retain(|virt_server| virt_server.read().unwrap().rpc_id != rpc_id);
        
        log::trace!("Virt servers after deallocation: {:?}", server_node.virt_servers);
        self.update_server_node(server_node);
        Ok(())
    }

    pub fn change_resource_configurations(&self, server_ip: &String, rpc_id: u64, compute_units: u32, memory: u64) -> Result<(),String> {
        let server_node = self.get_server_node(server_ip);

        if server_node.is_none() {
            log::error!("Server node not found: {}", server_ip);
            return Err("Server node not found".to_string());
        }

        let mut server_node = server_node.unwrap();

        let target_vserver = server_node.virt_servers.iter_mut().find(|virt_server| virt_server.read().unwrap().rpc_id == rpc_id);

        if target_vserver.is_none() {
            log::error!("Virt server not found: {}", rpc_id);
            return Err("Virt server not found".to_string());
        }

        let target_vserver = target_vserver.unwrap();
        let mut target_vserver_write_guard = target_vserver.write().unwrap();

        let tgpu = target_vserver_write_guard.gpu.clone();
        let mut gpu = tgpu.write().unwrap();

        let compute_units_diff = compute_units - target_vserver_write_guard.compute_units;
        let memory_diff = memory - target_vserver_write_guard.memory;

        if compute_units_diff > gpu.compute_units - gpu.allocated_compute_units || memory_diff > gpu.memory - gpu.allocated_memory {
            log::error!("Not enough resources to allocate");
            return Err("Not enough resources to allocate".to_string());
        }

        // call the server node

        stream_write!(get_writer!(server_node), format!("{}\n{},{},{}\n", FlytApiCommand::RMGR_SNODE_CHANGE_RESOURCES, rpc_id, compute_units, memory));

        let response = stream_read_response!(get_reader!(server_node), 2);

        if response[0] != "200" {
            log::error!("RMGR_SNODE_CHANGE_RESOURCES, Status: {}\n{}", response[0], response[1]);
            return Err(format!("RMGR_SNODE_CHANGE_RESOURCES, Status: {}\n{}", response[0], response[1]));
        }

        gpu.allocated_compute_units += compute_units_diff;
        gpu.allocated_memory += memory_diff;

        target_vserver_write_guard.compute_units = compute_units;
        target_vserver_write_guard.memory = memory;

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

