////use std::sync::{Arc, Mutex};

use std::{thread, io::BufReader, net::{TcpStream, TcpListener} };
use std::time::Duration;

use crate::{client_handler::FlytClientManager, common::{api_commands:: MetricsCommand, utils::StreamUtils}, servernode_handler::ServerNodesManager};

const UPSCALE_COUNT: u32 = 3;
const DOWNSCALE_COUNT: u32 = 3;

enum ChangeScale {
    ScaleUp,
    ScaleDown,
    ScaleNone,
}

pub struct MetricsHandler<'a> {
    client_mgr: &'a FlytClientManager<'a>,
    server_nodes_manager: &'a ServerNodesManager<'a>,
}

impl <'a> MetricsHandler<'a> {
    pub fn new(client_mgr: &'a FlytClientManager, server_nodes_manager: &'a ServerNodesManager) -> Self {
        MetricsHandler {
            client_mgr,
            server_nodes_manager,
        }
    }

    pub fn start_metrics_handler(&self, port: u16) {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    self.handle_metrics_request(stream)
                }
                Err(e) => {
                    log::error!("Error accepting connection: {}", e)
                }
            }
        }
    }

    pub fn start_period_handler(&self, interval_secs: u64) {
        loop {
            // Sleep for the specified interval
            thread::sleep(Duration::from_secs(interval_secs));

            // Perform the network operation
            self.handle_heartbeat();
        }
    }

    fn handle_heartbeat(&self) {
        // For all server nodes, get the heartbeat metrics.
        // Compute scale up or down and take action.
        let clients = self.client_mgr.get_all_clients();
        //let server_nodes = self.server_nodes_manager.get_all_server_nodes();
        for client_node in clients {
            if *client_node.is_migrating.read().unwrap() == false  && client_node.virt_server.is_some() {
                let mut virt_server = client_node.virt_server.as_ref().unwrap().write().unwrap();
                //let virt_server = virt_server.read().unwrap();
                let metrics = self.server_nodes_manager.get_server_node_metrics(&virt_server.ipaddr, virt_server.rpc_id);
                log::info!("Heart beat for server {} is {:?}", virt_server.ipaddr, metrics);

                let cur_compute = virt_server.compute_units;
                //let cur_mem = virt_server.memory;
                let mut usage = cur_compute;
                if let Some(metrics_vec) = metrics.as_ref() {
                    usage = if metrics_vec[0] % 2 == 0 {
                                metrics_vec[0]
                            } else {
                                metrics_vec[0] + 1
                            };
                }

                /* Not required now 
                let mut scaleflag = ChangeScale::ScaleNone;

                if usage > cur_compute {
                    // Scale up
                    scaleflag = ChangeScale::ScaleUp;
                }
                else if usage < cur_compute {
                    // Scale down
                    scaleflag = ChangeScale::ScaleDown;
                }

                let ret = match scaleflag {
                    ChangeScale::ScaleUp => self.server_nodes_manager.change_resource_configurations(&virt_server.ipaddr, virt_server.rpc_id, usage as u32, cur_mem),
                    ChangeScale::ScaleDown => self.server_nodes_manager.change_resource_configurations(&virt_server.ipaddr, virt_server.rpc_id, usage as u32, cur_mem),
                    ChangeScale::ScaleNone => todo!()
                };
                if ret.is_ok() {
                    log::info!("Resource updated successfully");
                }
                else {
                    log::error!("Error updating resource: initiating automigrate{:?}", ret);

                    let ret_val = self.server_nodes_manager.migrate_virt_server_auto(
                        self.client_mgr, 
                        &client_node.ipaddr.to_string(),
                        usage,
                        cur_mem);
                }
                */
                virt_server.actual_units = usage;
                log::info!("Updated aacutal units: {} for virt-server : {}", usage, virt_server.ipaddr);
            }

            // TODO: implement sandpiper paper logic
        }
    }

    fn handle_metrics_request(&self, stream: TcpStream) {
        let reader_clone = match stream.try_clone() {
            Ok(stream) => stream,
            Err(e) => {
                log::error!("Error cloning stream in metrics request: {}", e);
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
        
        let scalevalue_str = match StreamUtils::read_line(&mut reader) {
            Ok(scalevalue_str) => scalevalue_str,
            Err(e) => {
                log::error!("Error reading scalevalue: {}", e);
                return;
            }
        };

        let scalevalue = scalevalue_str.parse::<u32>().unwrap();
        let vm_ip = stream.peer_addr().unwrap().ip().to_string();

        log::info!("Scale received command: {} scalevalue: {} vm ip : {}", command.as_str(), scalevalue, vm_ip);

        match command.as_str() {
            MetricsCommand::CLIENTD_MMGR_UPSCALE => {
                self.server_scaleup(&vm_ip, scalevalue, ChangeScale::ScaleUp);
            }
            MetricsCommand::CLIENTD_MMGR_DOWNSCALE => {
                self.server_scaleup(&vm_ip, scalevalue, ChangeScale::ScaleDown);
            }
            _ => {
                log::error!("Invalid command: {}", command);
            }
        }
    }

    fn server_scaleup(&self, vm_ip: &String, scale_value: u32, scaleflag: ChangeScale) {
        let client = self.client_mgr.get_client(vm_ip);
        if client.is_none() {
            log::error!("Client VM {} not found", vm_ip);
            return;
        }

        let mut client = client.unwrap();
        
        if client.virt_server.is_none() {
            log::error!("VM {} is not running on any server node", vm_ip);
            return;
        }

        let (virt_server_ip, virt_server_rpc_id, cur_compute, cur_mem, _actual_units) = {
            let virt_server = client.virt_server.as_ref().unwrap().read().unwrap();
            (virt_server.ipaddr.clone(), virt_server.rpc_id, virt_server.compute_units, virt_server.memory, virt_server.actual_units)
        };

        let mut new_compute = match scaleflag {
            ChangeScale::ScaleUp => scale_value as u32, //(cur_compute * (100 + scale_value))/100 as u32,
            ChangeScale::ScaleDown => scale_value as u32, //(cur_compute * (100 - scale_value))/100 as u32,
            ChangeScale::ScaleNone => 0 as u32
        };

        if new_compute > 0 && new_compute != cur_compute {
            let vm_required_resources = self.server_nodes_manager.vm_resource_getter.get_vm_required_resources(vm_ip);
            
            if vm_required_resources.is_none() {
                log::error!("Scaleup VM resources not found for client: {}", vm_ip);
                return;
            }

            let mut vm_required_resources = vm_required_resources.unwrap();
            
            vm_required_resources.compute_units = new_compute;
            vm_required_resources.memory = cur_mem;

            let target_gpu = self.server_nodes_manager.get_free_gpu(&vm_required_resources);
            log::info!("Scaleup required compute: {} memory: {}, target_gpcu {} ", new_compute, cur_mem, target_gpu.is_none());

            if target_gpu.is_none() {
                match self.server_nodes_manager.get_server_node(&virt_server_ip) {
                    Some(server_node) => {
                        let mut max_available_compute = 0;

                        for gpu in server_node.gpus.iter() {
                            let gpu_read = gpu.read().unwrap();
                            max_available_compute = max_available_compute.max(gpu_read.compute_units);
                        }

                        // Now compare with new_compute
                        let update_compute = if new_compute <= max_available_compute {
                            new_compute
                        } else {
                            max_available_compute
                        };

                        log::info!("Scale-up expected {} but can only do {}", new_compute, update_compute);
                        new_compute = update_compute;
                    }
                    None => {
                        log::warn!("Scaling up No server node found for IP: {}", virt_server_ip);
                    }
                }
            }
            log::info!("Scaleup requested compute: {} memory: {}", new_compute, cur_mem);

            let ret = self.server_nodes_manager.change_resource_configurations(&virt_server_ip, virt_server_rpc_id, new_compute, cur_mem);
            if ret.is_ok() {
                log::info!("Resource updated successfully new_sm {} cur_sm {}", new_compute, cur_compute);
            }
            else {

                *client.is_migrating.get_mut().unwrap() = true;
                let ret_val = self.server_nodes_manager.migrate_virt_server_auto(
                        self.client_mgr, 
                        &client.ipaddr.to_string(),
                        new_compute,
                        cur_mem);
                *client.is_migrating.get_mut().unwrap() = false;
                log::info!("migrating resource: {:?}", ret);
            }
        }
    }

    /*
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

    fn migrate_vm_auto(&self, mut stream: UnixStream, mut reader: BufReader<UnixStream>) {

        log::info!("Received migrate auto request");

        let request_params = match StreamUtils::read_response(&mut reader, 1) {
            Ok(params) => params,
            Err(e) => {
                log::error!("Error reading request params: {}", e);
                return;
            }
        };
        
        let parts: Vec<&str> = request_params[0].split(',').collect();
        if parts.len() != 3 {
            log::error!("Invalid arguments for migrate command: {:?}", parts);
            let _ = StreamUtils::write_all(&mut stream, "400\nInvalid arguments\n".to_string());
            return;
        }

        let ipaddr = parts[0];

        let new_server_compute_units = match parts[1].parse::<u32>() {
            Ok(sm_cores) => sm_cores,
            Err(e) => {
                log::error!("Error parsing sm_cores: {}", e);
                let _ = StreamUtils::write_all(&mut stream, "400\nInvalid arguments\n".to_string());
                return;
            }
        };

        let new_server_memory = match parts[2].parse::<u64>() {
            Ok(memory) => memory,
            Err(e) => {
                log::error!("Error parsing memory: {}", e);
                let _ = StreamUtils::write_all(&mut stream, "400\nInvalid arguments\n".to_string());
                return;
            }
        };

        log::info!("Migrating VM: {}", ipaddr);
        let res = self.server_nodes_manager.migrate_virt_server_auto(
            self.client_mgr, 
            &ipaddr.to_string(),
            new_server_compute_units,
            new_server_memory);
        
        if res.is_err() {
            log::error!("Error migrating VM: {}", res.clone().unwrap_err());
            
            // resume the client if stopped
            let _ = self.client_mgr.resume_client(ipaddr);

            let _ = StreamUtils::write_all(&mut stream, format!("500\n{}\n", res.unwrap_err()));
        }
        else {
            log::info!("VM migrated successfully to server: {}", res.unwrap().read().unwrap().ipaddr );
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
    */
    
}
