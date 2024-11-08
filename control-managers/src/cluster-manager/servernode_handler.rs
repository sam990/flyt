

use log::info;

use crate::bookkeeping::*;
use crate::client_handler::FlytClientManager;
use crate::common::api_commands::FlytApiCommand;
use crate::common::types::StreamEnds;
use crate::common::utils::StreamUtils;

use std::sync::atomic::{AtomicI32, Ordering};
use std::collections::HashMap;
use std::{fs, thread};
use std::io::{ BufReader, Write };
use std::net::{TcpListener, TcpStream};
use std::path::Path;
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
    // Match for the case where you need to return None
    ($stream:expr, $data:expr, None) => {{
        match $stream.write_all($data.as_bytes()) {
            Ok(_) => {},
            Err(e) => {
                log::error!("Error writing to stream: {}", e);
                return None;
            }
        }
    }};
    
    // Default case: return an error message
    ($stream:expr, $data:expr) => {{
        match $stream.write_all($data.as_bytes()) {
            Ok(_) => {},
            Err(e) => {
                log::error!("Error writing to stream: {}", e);
                return Err(format!("Error writing to stream: {}", e));
            }
        }
    }};
}

macro_rules! stream_read_line {
    // Match for the case where you need to return None
    ($stream:expr, None) => {{
        match StreamUtils::read_line(&mut $stream) {
            Ok(data) => data,
            Err(e) => {
                log::error!("Error reading from stream: {}", e);
                return None;
            }
        }
    }};
    // Default case: return a specific error value
    ($stream:expr) => {{
        match StreamUtils::read_line(&mut $stream) {
            Ok(data) => data,
            Err(e) => {
                log::error!("Error reading from stream: {}", e);
                return Err(format!("Error reading from stream: {}", e));
            }
        }
    }};
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

const DEFAULT_ADD_LOAD: i32 = 5;

fn split_u32(value: u32) -> (u16, u16) {
    let lower = (value & 0xFFFF) as u16;      // Extract the lower 16 bits
    let upper = (value >> 16) as u16;         // Extract the upper 16 bits
    (lower, upper)
}

pub struct ServerNodesManager<'a> {
    server_nodes: Mutex<HashMap<String, ServerNode>>,
    pub vm_resource_getter: &'a VMResourcesGetter,
    server_index: AtomicI32,
}

impl<'a> ServerNodesManager<'a> {

    pub fn new( resource_getter: &'a VMResourcesGetter ) -> Self {
        ServerNodesManager {
            server_nodes: Mutex::new(HashMap::new()),
            vm_resource_getter: resource_getter,
            server_index: AtomicI32::new(0),
        }
    }

    // Atomically gets the current counter value and then increments it
    fn get_next_server_index(&self) -> i32 {
        let mut current_index = self.server_index.load(Ordering::SeqCst);
        let max_value = self.server_nodes.lock().unwrap().len() as i32;
        let next_index = (current_index + 1) % max_value;

        // Atomically store the new value
        self.server_index.store(next_index, Ordering::SeqCst);
        next_index
    }

    fn set_next_server_index(&self, index: i32) {
        self.server_index.store(index, Ordering::SeqCst);
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

        if self.exists(&server_ip) {
            log::info!("Server node already exists: {}", server_ip);
            
            let mut server_node = self.get_server_node(&server_ip).unwrap();
            server_node.stream.write().unwrap().writer = stream;
            server_node.stream.write().unwrap().reader = reader;
            server_node.virt_servers = Vec::new();
            server_node.gpus = Vec::new();

            self.update_server_node(server_node);
        }
        else {
            
            let server_node = ServerNode {
                ipaddr: server_ip.clone(),
                gpus: Vec::new(),
                stream: Arc::new(RwLock::new(StreamEnds{writer: stream, reader })),
                virt_servers: Vec::new(),
                metrics: None,
                add_load: DEFAULT_ADD_LOAD,
            };
        
            self.add_server_node(server_node);
        }

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
            log::info!("Server node gpu string: {}", gpu_info_str);
            gpus.push(gpu);
        }

        server_node.gpus = gpus;
        self.update_server_node(server_node);
        log::info!("Server node gpus updated: {}", server_node_ip );
        Ok(())
    }

    pub fn get_free_gpu(&self, required_resources: &VMResources, client_id: i32) -> Option<(String, u64)> {
        /*
        let host_server_node = self.get_server_node(&required_resources.host_ip);
        
        if host_server_node.is_some() {
            let host_server_node = host_server_node.unwrap();
            let gpu_id = check_resource_availability(&host_server_node, required_resources);
            if gpu_id.is_some() {
                return Some((host_server_node.ipaddr.clone(), gpu_id.unwrap()));
            }
        }

        // If not, then checking all other servers

        let server_nodes = self.get_all_server_nodes();
        
        for server_node in server_nodes {
            let gpu_id = check_resource_availability(&server_node, required_resources);
            if gpu_id.is_some() {
                return Some((server_node.ipaddr.clone(), gpu_id.unwrap()));
            }
        }
        */
        let server_nodes = self.get_all_server_nodes();
        let total_nodes = server_nodes.len();
        let server_index = self.get_next_server_index();

        // Iterate starting from `start_index` and wrapping around to form a circular search
        for i in 0..total_nodes {
            // Calculate the current index, wrapping around using modulo
            let index = (server_index as usize + i) % total_nodes;
            log::info!("client_id index in get_free_fpu {} i: {} total: {}", index, i, total_nodes);

            // Check the current server node for available GPU resources
            let server_node = &server_nodes[index];
            if let Some(gpu_id) = check_resource_availability(&server_node, required_resources) {
                self.set_next_server_index(index as i32);
                return Some((server_node.ipaddr.clone(), gpu_id));
            }
        }
            log::info!("could not find server for client {}", client_id);

        None
    
    }

    pub fn get_max_gpu(&self, required_resources: &VMResources) -> (u64, u64) {
        /*
        let host_server_node = self.get_server_node(&required_resources.host_ip);
        
        if host_server_node.is_some() {
            let host_server_node = host_server_node.unwrap();
            let gpu_id = check_resource_availability(&host_server_node, required_resources);
            if gpu_id.is_some() {
                return Some((host_server_node.ipaddr.clone(), gpu_id.unwrap()));
            }
        }

        // If not, then checking all other servers

        let server_nodes = self.get_all_server_nodes();
        
        for server_node in server_nodes {
            let gpu_id = check_resource_availability(&server_node, required_resources);
            if gpu_id.is_some() {
                return Some((server_node.ipaddr.clone(), gpu_id.unwrap()));
            }
        }
        */
        let server_nodes = self.get_all_server_nodes();
        let total_nodes = server_nodes.len();
        let mut max_available_compute = 0;
        let mut max_available_memory = 0;

        // Iterate starting from `start_index` and wrapping around to form a circular search
        for i in 0..total_nodes {

            // Check the current server node for available GPU resources
            let server_node = &server_nodes[i];
            let (server_compute, server_memory) = max_resource_availability(&server_node);

            let max_compute = max_available_compute.max(server_compute);
            //TODO: handle memory as well
            if max_compute == server_compute {
                max_available_compute = server_compute;
                max_available_memory = server_memory;
            }
        }
        (max_available_compute, max_available_memory)
    }

    pub fn allocate_vm_resources(&self, client_ip: &String, client_id: i32) -> Result<Arc<RwLock<VirtServer>>,String> {
        let vm_required_resources = self.vm_resource_getter.get_vm_required_resources(client_ip);
        
        log::info!("Allocating VM resources for client: {} {}", client_ip, client_id);
        log::info!("VM resources required: {:?}", vm_required_resources);

        if vm_required_resources.is_none() {
            log::error!("VM resources not found for client: {} {}", client_ip, client_id);
            return Err("VM resources not found".to_string());
        }

        let vm_required_resources = vm_required_resources.unwrap();

        let target_gpu = self.get_free_gpu(&vm_required_resources, client_id);

        log::info!("after get_free_gpu");
        if target_gpu.is_none() {
            log::error!("No free GPU found for client: {}", client_ip);
            return Err("No free GPU found".to_string());
        }

        let (target_server_ip, target_gpu_id) = target_gpu.unwrap();
        
        let virt_server = self.create_virt_server(&target_server_ip, target_gpu_id, vm_required_resources.compute_units, vm_required_resources.memory, true);
        log::info!("after create_virt_server");

        if virt_server.is_err() {
            log::error!("Error creating virt server for client: {} {}", client_ip, client_id);
            return Err("Error creating virt server".to_string());
        }

        let virt_server = virt_server.unwrap();
        log::info!("Allocated for VM: {} client_id: {} GPU server: {}  gpu_id {}", client_ip, client_id, target_server_ip, target_gpu_id);

        Ok(virt_server)

    }

    pub fn create_virt_server(&self, snode_ip: &String, gpu_id: u64, compute_units: u32, memory: u64, allow_overprovision: bool) -> Result<Arc<RwLock<VirtServer>>,String> {
        
        let server_node = self.get_server_node(&snode_ip);

        // For all server nodes, get the heartbeat metrics.
        // Compute scale up or down and take action.
        if server_node.is_none() {
            log::error!("Server node not found: {}", snode_ip);
            return Err("Server node not found".to_string());
        }

        let mut server_node = server_node.unwrap();

        let target_gpu = server_node.gpus.iter_mut().find(|gpu| gpu.read().unwrap().gpu_id == gpu_id);

        if target_gpu.is_none() {
            log::error!("GPU not found: {}", gpu_id);
            return Err("GPU not found".to_string());
        }

        let target_gpu = target_gpu.unwrap();

        {
            let gpu_read = target_gpu.read().unwrap();

            if !allow_overprovision && (compute_units > (gpu_read.compute_units - gpu_read.allocated_compute_units) 
                                        || memory > (gpu_read.memory - gpu_read.allocated_memory)) 
            {
                log::error!("Not enough resources to allocate compute_units: {}, memory: {}", compute_units, memory);
                log::error!("Available compute_units: {}, memory: {}", gpu_read.compute_units - gpu_read.allocated_compute_units, gpu_read.memory - gpu_read.allocated_memory);
                return Err("Not enough resources to allocate".to_string());
            }
        }

        stream_write!(get_writer!(server_node), format!("{}\n{},{},{}\n", FlytApiCommand::RMGR_SNODE_ALLOC_VIRT_SERVER, gpu_id, compute_units, memory));

        let response = stream_read_response!(get_reader!(server_node), 2);

        if response[0] != "200" {
            log::error!("RMGR_SNODE_ALLOC_VIRT_SERVER, Status: {}\n{}", response[0], response[1]);
            return Err(format!("RMGR_SNODE_ALLOC_VIRT_SERVER, Status: {}\n{}", response[0], response[1]));
        }

        let virt_server_rpc_id = response[1].parse::<u64>().unwrap();

        let virt_server = Arc::new(RwLock::new(VirtServer {
            ipaddr: server_node.ipaddr.clone(),
            compute_units,
            memory,
            rpc_id: virt_server_rpc_id,
            actual_units: compute_units,
            gpu: target_gpu.clone(),
        }));

        server_node.virt_servers.push(virt_server.clone());

        {
            let mut gpu_write = target_gpu.write().unwrap();
            gpu_write.allocated_compute_units += compute_units;
            gpu_write.allocated_memory += memory;
        }

        self.update_server_node(server_node);

        Ok(virt_server)
    }

    pub fn checkpoint(&self, virt_ip: &String, rpc_id: u64, ckp_path: &String) -> Result<(),String> {
        let server_node = self.get_server_node(&virt_ip);

        if server_node.is_none() {
            log::error!("Server node not found: {}", virt_ip);
            return Err("Server node not found".to_string());
        }

        let server_node = server_node.unwrap();

        let target_vserver = server_node.virt_servers.iter().find(|virt_server| virt_server.read().unwrap().rpc_id == rpc_id);

        if target_vserver.is_none() {
            log::error!("Virt server not found: {}", rpc_id);
            return Err("Virt server not found".to_string());
        }

        stream_write!(get_writer!(server_node), format!("{}\n{}\n{}\n", FlytApiCommand::RMGR_SNODE_CHECKPOINT, rpc_id, ckp_path));

        let response = stream_read_response!(get_reader!(server_node), 2);

        if response[0] != "200" {
            log::error!("RMGR_SNODE_CHECKPOINT, Status: {}\n{}", response[0], response[1]);
            return Err(format!("RMGR_SNODE_CHECKPOINT, Status: {}\n{}", response[0], response[1]));
        }

        Ok(())
    }

    pub fn restore_state(&self, snode_ip: &String, rpc_id: u64, ckp_path: &String) -> Result<(),String> {
        
        let server_node = self.get_server_node(&snode_ip);

        if server_node.is_none() {
            log::error!("Server node not found: {}", snode_ip);
            return Err("Server node not found".to_string());
        }

        let server_node = server_node.unwrap();

        stream_write!(get_writer!(server_node), format!("{}\n{}\n{}\n", FlytApiCommand::RMGR_SNODE_RESTORE, rpc_id, ckp_path));

        let response = stream_read_response!(get_reader!(server_node), 2);

        if response[0] != "200" {
            log::error!("RMGR_SNODE_RESTORE, Status: {}\n{}", response[0], response[1]);
            return Err(format!("RMGR_SNODE_RESTORE, Status: {}\n{}", response[0], response[1]));
        }

        Ok(())
    }

    pub fn migrate_virt_server(&self, client_mgr: &FlytClientManager, client_ip: &String, client_id: i32, target_snode_id: &String, target_gpu_id: u64, new_sm_cores: u32, new_mem: u64) -> Result<Arc<RwLock<VirtServer>>,String> {
        
        log::info!("Migrating client vm: {},{} to server node: {}", client_ip, client_id, target_snode_id);

        match client_mgr.stop_client(client_ip, client_id) {
            Ok(_) => {
                log::info!("Client VM {},{} stopped successfully", client_ip, client_id);
            }
            Err(e) => {
                log::error!("Error stopping client VM {},{}: {}", client_ip, client_id, e);
                return Err(format!("Error stopping client VM {},{}: {}", client_ip, client_id, e));
            }
        }

        let ckp_base_path = get_ckp_base_path();

        if ckp_base_path.is_none() {
            log::error!("Checkpoint base path not found");
            return Err("Checkpoint base path not found".to_string());
        }

        let ckp_base_path = ckp_base_path.unwrap();
        let ckp_path = format!("{}/{}", ckp_base_path, client_ip);

        log::debug!("Checkpoint path: {}", ckp_path);

        // if path exists, remove it
        if Path::new(&ckp_path).exists() {
            let res = fs::remove_dir_all(&ckp_path);
            if res.is_err() {
                log::error!("Error removing path: {}", res.err().unwrap());
                return Err("Error clearing checkpoint path".to_string());
            }
        }

        // create new path
        if fs::create_dir_all(&ckp_path).is_err() {
            log::error!("Error creating path: {}", ckp_path);
            return Err(format!("Error creating path: {}", ckp_path));
        }

        let client = client_mgr.get_client(client_ip, client_id);
        if client.is_none() {
            log::error!("Client is removed or terminated {}, {}", client_ip, client_id);
            return Err("client not found".to_string());
        }

        let client = client.unwrap();

        let snode_ip = client.virt_server.as_ref().unwrap().read().unwrap().ipaddr.clone();
        let rpc_id = client.virt_server.as_ref().unwrap().read().unwrap().rpc_id;


        let vserver = thread::scope( |s| {
            
            let checkpoint_thread = s.spawn( || {
                let res = self.checkpoint(&snode_ip, rpc_id, &ckp_path);
                if res.is_err() {
                    log::error!("Error checkpointing virt server: {}", res.err().unwrap());
                    return Err(format!("Error checkpointing virt server"));
                }
                Ok(())
            });

            let create_vserver_thread = s.spawn(|| {
                let res = self.create_virt_server(target_snode_id, target_gpu_id, new_sm_cores, new_mem, true);
                if res.is_err() {
                    log::error!("Error allocating and restoring virt server: {}", res.err().unwrap());
                    return Err(format!("Error allocating and restoring virt server"));
                }
                Ok(res.unwrap())
            });

            let vserver = match create_vserver_thread.join() {
                Ok(res) => res,
                Err(e) => {
                    log::error!("Error creating virt server: {:?}", e);
                    return Err(format!("Error creating virt server"));
                }
            };

            let vserver = match vserver {
                Ok(vserver) => vserver,
                Err(e) => {
                    log::error!("Error creating virt server: {}", e);
                    return Err(format!("Error creating virt server: {}", e));
                }
            };

            let checkpoint_thread_res = checkpoint_thread.join();


            let ckp_err_handler = |e: String| {
                log::error!("Error checkpointing virt server: {:?}", e);

                // free the allocated resources
                let res = self.free_virt_server(target_snode_id, vserver.read().unwrap().rpc_id, true);
                
                if res.is_err() {
                    log::error!("Error freeing virt server: {}", res.err().unwrap());
                }
            };


            if checkpoint_thread_res.is_err() {
                ckp_err_handler("thread join error".to_string());
                return Err("Error checkpointing virt server".to_string());
            }

            let checkpoint_thread_res = checkpoint_thread_res.unwrap();

            if checkpoint_thread_res.is_err() {
                ckp_err_handler(checkpoint_thread_res.err().unwrap());
                return Err("Error checkpointing virt server".to_string());
            }

            Ok(vserver)
        });

        if vserver.is_err() {
            log::error!("Error creating/checkpoint virt server: {}", vserver.err().unwrap());
            return Err(format!("Error creating/checkpoint virt server"));
        }
        
        let vserver = vserver.unwrap();
        
        let new_server_ip = vserver.read().unwrap().ipaddr.clone();
        let new_rpc_id = vserver.read().unwrap().rpc_id;

        // restore the state
        let res = self.restore_state(&new_server_ip, new_rpc_id, &ckp_path);
        if res.is_err() {
            log::error!("Error restoring virt server: {}", res.err().unwrap());
            return Err(format!("Error restoring virt server"));
        }

        
        let res = client_mgr.change_virt_server(&client_ip, client_id, &vserver);
        
        if res.is_err() {
            log::error!("Error changing virt server: {}", res.err().unwrap());
            return Err(format!("Error changing virt server for client"));
        }

        let res = client_mgr.resume_client(&client_ip, client_id);
        
        if res.is_err() {
            log::error!("Error resuming client VM: {}", res.err().unwrap());
            return Err(format!("Error resuming client VM"));
        }

        log::info!("VM migrated successfully");

        let res = self.free_virt_server(&snode_ip, rpc_id, true);

        if res.is_err() {
            log::error!("Error freeing virt server: {}", res.err().unwrap());
        }

        Ok(vserver)
    
    }

    pub fn migrate_virt_server_auto(&self, client_mgr: &FlytClientManager, client_ip: &String, client_id: i32, new_sm_cores: u32, new_mem: u64) -> Result<Arc<RwLock<VirtServer>>,String> {
        let vm_required_resources = self.vm_resource_getter.get_vm_required_resources(client_ip);
        
        if vm_required_resources.is_none() {
            log::error!("VM resources not found for client: {}", client_ip);
            return Err("VM resources not found".to_string());
        }

        let mut vm_required_resources = vm_required_resources.unwrap();
        
        if vm_required_resources.compute_units  == new_sm_cores {
            log::error!("VM resources not changed: {}", new_sm_cores);
            return Err("No change in VM resources ".to_string());
        }

        vm_required_resources.compute_units = new_sm_cores;
        vm_required_resources.memory = new_mem;

        let target_gpu = self.get_free_gpu(&vm_required_resources, client_id);

        if target_gpu.is_none() {
            log::error!("No free GPU found for client: {}", client_ip);
            return Err("No free GPU found".to_string());
        }

        let (target_server_ip, target_gpu_id) = target_gpu.unwrap();

        let client = client_mgr.get_client(client_ip, client_id);
        if client.is_none() {
            log::error!("Client is removed or terminated {}, {}", client_ip, client_id);
            return Err("client not found".to_string());
        }

        let client = client.unwrap();

        let snode_ip = client.virt_server.as_ref().unwrap().read().unwrap().ipaddr.clone();
        if target_server_ip == snode_ip {
            log::error!("Migrating to same ip address: {}", target_server_ip);
            return Err("No new GPU found".to_string());
        }

        self.migrate_virt_server(client_mgr, client_ip, client_id, &target_server_ip, target_gpu_id, new_sm_cores, new_mem)
        
    }

    pub fn free_virt_server(&self, virt_ip: &String, rpc_id: u64, sendtoserver: bool) -> Result<(),String> {

        info!("Deallocating virt server: {}:{}", virt_ip, rpc_id);

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

        if sendtoserver == true {
            log::trace!("Sending dealloc command to server node: {}/{}", virt_ip, rpc_id);
            stream_write!(get_writer!(server_node), format!("{}\n{}\n", FlytApiCommand::RMGR_SNODE_DEALLOC_VIRT_SERVER, rpc_id));
            
            let response = stream_read_response!(get_reader!(server_node), 2);

            log::trace!("Response from server node {} for deallocate: {:?}", virt_ip, response);

            if response[0] != "200" {
                log::error!("RMGR_SNODE_DEALLOC_VIRT_SERVER, Status: {}\n{}", response[0], response[1]);
                return Err(format!("RMGR_SNODE_DEALLOC_VIRT_SERVER, Status: {}\n{}", response[0], response[1]));
            }
        }

        let target_vserver = target_vserver.unwrap();

        {
            let target_vserver_lock_guard = target_vserver.read().unwrap();
            let mut gpu_write_lock_guard = target_vserver_lock_guard.gpu.write().unwrap();

            gpu_write_lock_guard.allocated_compute_units -= target_vserver_lock_guard.compute_units;
            gpu_write_lock_guard.allocated_memory -= target_vserver_lock_guard.memory;
        }

        server_node.virt_servers.retain(|virt_server| virt_server.read().unwrap().rpc_id != rpc_id);
        
        log::debug!("Virt servers after deallocation" );
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

        let mut total_actual_units = 0;

        for virt_server in &server_node.virt_servers {
            let virt_server_guard = virt_server.read().unwrap();  // Lock the RwLock for reading
            if virt_server_guard.actual_units == 0 {
                total_actual_units += virt_server_guard.compute_units;
            }
            else {
                total_actual_units += virt_server_guard.actual_units;
            }
        }



        let target_vserver = server_node.virt_servers.iter_mut().find(|virt_server| virt_server.read().unwrap().rpc_id == rpc_id);

        if target_vserver.is_none() {
            log::error!("Virt server not found: {}", rpc_id);
            return Err("Virt server not found".to_string());
        }

        let target_vserver = target_vserver.unwrap();
        let mut target_vserver_write_guard = target_vserver.write().unwrap();

        let tgpu = target_vserver_write_guard.gpu.clone();
        let mut gpu = tgpu.write().unwrap();

        // Calculate the difference between GPU compute_units and total_actual_units
        let available_units = gpu.compute_units as i32  + target_vserver_write_guard.actual_units as i32 - total_actual_units as i32;

        let compute_units_diff = compute_units - target_vserver_write_guard.compute_units;
        let memory_diff = memory - target_vserver_write_guard.memory;

        //if compute_units_diff + gpu.allocated_compute_units > gpu.compute_units|| memory_diff + gpu.allocated_memory > gpu.memory {
        //if ((compute_units as i32) > available_units) || (memory_diff + gpu.allocated_memory > gpu.memory)  {
        // with overprovision
        if (compute_units > gpu.compute_units) || (memory_diff + gpu.allocated_memory > gpu.memory)  {
            log::error!("Not enough resources to allocate compute_units: {}, memory: {}", compute_units, memory);
            log::error!("Available compute_units: {}, memory: {}", gpu.compute_units - gpu.allocated_compute_units, gpu.memory - gpu.allocated_memory);
            log::error!("Current compute_units: {}, memory: {}", target_vserver_write_guard.compute_units, target_vserver_write_guard.memory);
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

    pub fn get_server_node_metrics(&self, server_node_ip: &String , rpc_id: u64) -> Option<Vec<u32>> {

        log::info!("Getting metrics for servernode: {}", server_node_ip);

        // Check if the server node exists
        if !self.exists(server_node_ip) {
            log::error!("Server node not found: {}", server_node_ip);
            return None;
        }

        let server_node = match self.get_server_node(server_node_ip) {
            Some(node) => node,
            None => {
                log::error!("Failed to retrieve server node: {}", server_node_ip);
                return None;
            }
        };

        let mut results = Vec::with_capacity(3); // usage, call-rate, avg-latency Throughput, threads rate, wrap %, avg cycle activity, total instructions executed

        // Send throughput metrics request
        stream_write!(get_writer!(server_node), format!("{}\n{}\n", FlytApiCommand::SNODE_SEND_METRICS_THROUGHPUT, rpc_id), None);

        let status = stream_read_line!(get_reader!(server_node), None);
        log::info!("throughput status : {}", status);

        if status != "200" {
            let err_msg = stream_read_line!(get_reader!(server_node), None);
            log::error!("RMGR_SNODE_SEND_METRIC_THROUGHPUT failed, Status: {}\n{}", status, err_msg);
            return None;
        }
        
        let metrics = stream_read_line!(get_reader!(server_node), None);
        log::info!("throughput metrics : {}", metrics);
        // Step 1: Remove the square brackets
        let trimmed_str = metrics.trim_matches(|c| c == '[' || c == ']');

        // Step 2: Split the string by commas and collect the values into a vector
        let values: Vec<u32> = trimmed_str
            .split(',')
            .map(|s| s.trim().parse().expect("Failed to parse value"))
            .collect();
    
        // Step 3: Extract the values into individual variables (assuming exactly 3 values)
        if values.len() == 3 {
            results.extend_from_slice(&values);
            log::info!("Utilization values extracted: {}, {}, {}", values[0], values[1], values[2]);
        } else {
            log::error!("RMGR_SNODE_SEND_METRIC_UTILIZATION, expected 3 values but got: {}\n", values.len());
            return None;
        }
        return Some(results); // Return the results vector wrapped in Some
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

fn max_resource_availability(server_node: &ServerNode) -> (u64, u64) {
    let mut max_available_compute = 0 as u64;
    let mut max_available_memory = 0 as u64;

    for gpu in server_node.gpus.iter() {
        let gpu_read = gpu.read().unwrap();
        let remain_compute_units = gpu_read.compute_units - gpu_read.allocated_compute_units;
        let remain_memory = gpu_read.memory - gpu_read.allocated_memory;
        let max_compute = max_available_compute.max(remain_compute_units as u64);
        //TODO: handle memory as well
        if max_compute == u64::from(remain_compute_units) {
            max_available_compute = max_compute;
            max_available_memory = remain_memory as u64;
        }
    }
    (max_available_compute, max_available_memory)
}

