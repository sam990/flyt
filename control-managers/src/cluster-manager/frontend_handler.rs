use std::{fs, io::BufReader, os::unix::net::{UnixListener, UnixStream}, path::Path};

use crate::{client_handler::FlytClientManager, common::{api_commands::FrontEndCommand, utils::StreamUtils}, servernode_handler::ServerNodesManager};


#[derive(PartialEq, Debug)]
enum ChangeConfigFor {
    SmCores,
    Memory,
    Both,
}

pub struct FrontendHandler<'a> {
    client_mgr: &'a FlytClientManager<'a>,
    server_nodes_manager: &'a ServerNodesManager<'a>,
}

impl <'a> FrontendHandler<'a> {
    pub fn new(client_mgr: &'a FlytClientManager, server_nodes_manager: &'a ServerNodesManager) -> Self {
        FrontendHandler {
            client_mgr,
            server_nodes_manager,
        }
    }

    pub fn start_listening(&self, socket_path: &str) {

        if Path::new(socket_path).exists() {
            fs::remove_file(socket_path).unwrap();
        }

        let listener = UnixListener::bind(socket_path).unwrap();

        log::info!("Frontend handler listening on {}", socket_path);

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    self.handle_request(stream);
                }
                Err(e) => {
                    log::error!("Error: {}", e);
                    break;
                }
            }
        }
    }

    fn handle_request(&self, stream: UnixStream) {
        let reader_clone = match stream.try_clone() {
            Ok(stream) => stream,
            Err(e) => {
                log::error!("Error cloning stream: {}", e);
                return;
            }
        };
        let mut reader = BufReader::new(reader_clone);
        
        let command = match StreamUtils::read_line(&mut reader) {
            Ok(command) => command,
            Err(e) => {
                log::error!("Error reading command: {}", e);
                return;
            }
        };
        
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
                self.change_resources(stream, reader, ChangeConfigFor::SmCores);
            }
            FrontEndCommand::CHANGE_MEMORY => {
                self.change_resources(stream, reader, ChangeConfigFor::Memory);
            }
            FrontEndCommand::CHANGE_SM_CORES_AND_MEMORY => {
                self.change_resources(stream, reader, ChangeConfigFor::Both);
            }
            FrontEndCommand::MIGRATE_VIRT_SERVER => {
                self.migrate_vm(stream, reader);
            }
            _ => {
                log::error!("Invalid command");
            }
        }
    }

    fn migrate_vm(&self, mut stream: UnixStream, mut reader: BufReader<UnixStream>) {

        log::info!("Received migrate request");

        let request_params = match StreamUtils::read_response(&mut reader, 1) {
            Ok(params) => params,
            Err(e) => {
                log::error!("Error reading request params: {}", e);
                return;
            }
        };
        
        let parts: Vec<&str> = request_params[0].split(',').collect();
        if parts.len() != 5 {
            log::error!("Invalid arguments for migrate command: {:?}", parts);
            let _ = StreamUtils::write_all(&mut stream, "400\nInvalid arguments\n".to_string());
            return;
        }

        let ipaddr = parts[0];
        let new_server_ip = parts[1];
        let new_server_gpu_id = match parts[2].parse::<u64>() {
            Ok(gpu_id) => gpu_id,
            Err(e) => {
                log::error!("Error parsing gpu_id: {}", e);
                let _ = StreamUtils::write_all(&mut stream, "400\nInvalid arguments\n".to_string());
                return;
            }
        };

        let new_server_compute_units = match parts[3].parse::<u32>() {
            Ok(sm_cores) => sm_cores,
            Err(e) => {
                log::error!("Error parsing sm_cores: {}", e);
                let _ = StreamUtils::write_all(&mut stream, "400\nInvalid arguments\n".to_string());
                return;
            }
        };

        let new_server_memory = match parts[4].parse::<u64>() {
            Ok(memory) => memory,
            Err(e) => {
                log::error!("Error parsing memory: {}", e);
                let _ = StreamUtils::write_all(&mut stream, "400\nInvalid arguments\n".to_string());
                return;
            }
        };

        log::info!("Migrating VM: {} to server: {} with gpu_id: {}", ipaddr, new_server_ip, new_server_gpu_id);
        let res = self.server_nodes_manager.migrate_virt_server(
            self.client_mgr, 
            &ipaddr.to_string(),
            &new_server_ip.to_string(),
            new_server_gpu_id,
            new_server_compute_units,
            new_server_memory);
        
        if res.is_err() {
            log::error!("Error migrating VM: {}", res.clone().unwrap_err());
            
            // resume the client if stopped
            let _ = self.client_mgr.resume_client(ipaddr);

            let _ = StreamUtils::write_all(&mut stream, format!("500\n{}\n", res.unwrap_err()));
        }
        else {
            log::info!("VM migrated successfully");
            let _ = StreamUtils::write_all(&mut stream, "200\nVM migrated successfully\n".to_string());
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
                response.push_str(&format!("{},,,,,{}\n",
                    vm.ipaddr,
                    *vm.is_active.read().unwrap()
                ));
            }
            
        }
        let _ = StreamUtils::write_all(&mut stream, response);
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
        let _ = StreamUtils::write_all(&mut stream, response);
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
        let _ = StreamUtils::write_all(&mut stream, response);
    }

    fn change_resources(&self, mut stream: UnixStream, mut reader: BufReader<UnixStream>, change_for: ChangeConfigFor) {

        log::info!("Received change resource request");

        let buffer = match StreamUtils::read_response(&mut reader, 1) {
            Ok(buffer) => buffer,
            Err(e) => {
                log::error!("Error reading buffer: {}", e);
                return;
            }
        };
        let parts: Vec<&str> = buffer[0].split(',').collect();
        if (change_for != ChangeConfigFor::Both && parts.len() != 2) || (change_for == ChangeConfigFor::Both && parts.len() != 3 ){
            log::error!("Invalid arguments for change resource command {:?} {:?}", change_for, parts);
            let _ = StreamUtils::write_all(&mut stream, "400\nInvalid arguments\n".to_string());
            return;
        }
        let ipaddr = parts[0];
        let new_resource = parts[1].parse::<u64>().unwrap();

        log::info!("Changing resource for VM: {}, new resource: {} for {:?}", ipaddr, new_resource, change_for);

        let client = self.client_mgr.get_client(ipaddr);
        if client.is_none() {
            log::error!("Client VM {} not found", ipaddr);
            let _ = StreamUtils::write_all(&mut stream, "500\nVM not found\n".to_string());
            return;
        }

        let client = client.unwrap();
        
        if client.virt_server.is_none() {
            log::error!("VM {} is not running on any server node", ipaddr);
            let _ = StreamUtils::write_all(&mut stream, "500\nVM is not running on any server node\n".to_string());
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
            log::info!("Resource updated successfully");
            let _ = StreamUtils::write_all(&mut stream, "200\nResource updated successfully\n".to_string());
        }
        else {
            log::error!("Error updating resource: {:?}", ret);
            let _ = StreamUtils::write_all(&mut stream, format!("500\n{}\n", ret.unwrap_err()));
        }

    }
    
}