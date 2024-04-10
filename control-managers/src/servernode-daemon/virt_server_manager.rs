use std::{collections::HashMap, process::{Child, Command}, sync::{Arc, Mutex}, time::Duration};
use ipc_rs::MessageQueue;
use crate::common::{api_commands::FlytApiCommand, types::MqueueClientControlCommand, utils::Utils};

const PROJ_ID: i32 = 0x42;

#[derive(Clone)]
struct VirtServer {
    id: u64,
    gpu_id: u32,
    num_sm_cores: u32,
    gpu_memory: u64,
    process: Arc<Mutex<Child>>,
    send_id: i64,
    recv_id: i64,
}

pub struct VirtServerManager {
    counter: Mutex<u64>,
    virts_servers: Mutex<HashMap<u64,VirtServer>>,
    message_queue: MessageQueue,
    virt_server_program_path: String,
}



impl VirtServerManager {

    pub fn new(mqueu_path: &str, virt_server_program_path: String) -> VirtServerManager {

        let key = ipc_rs::PathProjectIdKey::new(mqueu_path.to_string(), PROJ_ID);
        let message_queue = MessageQueue::new(ipc_rs::MessageQueueKey::PathKey(key)).create().init().unwrap();
        

        VirtServerManager {
            counter: Mutex::new(0),
            virts_servers: Mutex::new(HashMap::new()),
            message_queue: message_queue,
            virt_server_program_path
        }
    }

    fn get_virt_server(&self, rpc_id: u64) -> Option<VirtServer> {
        let virt_servers = self.virts_servers.lock().unwrap();
        let virt_server = virt_servers.get(&rpc_id)?;
        Some(virt_server.clone())
    }

    fn update_virt_server(&self, rpc_id: u64, virt_server: VirtServer) {
        let mut virt_servers = self.virts_servers.lock().unwrap();
        virt_servers.insert(rpc_id, virt_server);
    }


    pub fn create_virt_server(&self, gpu_id: u32, gpu_memory: u64, num_sm_cores: u32, total_gpu_sm_cores: u32) -> Result<u64,String> {
        let rpc_id = {
            let mut counter = self.counter.lock().unwrap();
            *counter += 1;
            *counter
        };
        
        let gpu_cores_percentage = (num_sm_cores as f64 / total_gpu_sm_cores as f64).round() as u32;

        let send_id = rpc_id as i64;
        let recv_id = send_id << 32;

        let virt_server_process = Command::new(self.virt_server_program_path.as_str())
            .env("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25", gpu_cores_percentage.to_string())
            .arg(rpc_id.to_string())
            .arg(gpu_id.to_string())
            .arg(gpu_memory.to_string())
            .arg(num_sm_cores.to_string())
            .spawn()
            .map_err(|e| format!("Error starting virt server: {}", e))?;


        
        let virt_server = VirtServer {
            id: rpc_id,
            gpu_id: gpu_id,
            num_sm_cores: num_sm_cores,
            gpu_memory: gpu_memory,
            process: Arc::new(Mutex::new(virt_server_process)),
            send_id: send_id,
            recv_id: recv_id,
        };

        self.virts_servers.lock().unwrap().insert(rpc_id, virt_server);
        
        Ok(rpc_id)
        
    }

    pub fn remove_virt_server(&self, rpc_id: u64) -> Result<(),String> {
        let mut virt_servers = self.virts_servers.lock().unwrap();
        let virt_server = virt_servers.get(&rpc_id).ok_or("Virt server not found")?;
        
        virt_server.process.lock().unwrap().kill().map_err(|e| format!("Error killing virt server: {}", e))?;
        virt_servers.remove(&rpc_id);
        
        Ok(())
    }

    pub fn change_resources(&self, rpc_id: u64, new_num_sm_cores: u32, new_gpu_memory: u64) -> Result<(),String> {

        let mut virt_server = self.get_virt_server(rpc_id).ok_or("Virt server not found")?;
        let mqueue_cmd = MqueueClientControlCommand::new(FlytApiCommand::SNODE_VIRTS_CHANGE_RESOURCES, format!("{},{}", new_num_sm_cores, new_gpu_memory).as_str()).as_bytes();

        self.message_queue.send( &mqueue_cmd, virt_server.send_id).map_err(|e| format!("Error sending message to virt server: {}", e))?;

        let recv_bytes = self.message_queue.recv_type_timed(virt_server.recv_id, Duration::from_secs(5)).map_err(|e| format!("Error receiving message from virt server: {}", e))?;

        let recv_status = Utils::convert_bytes_to_u32(&recv_bytes).ok_or("Error converting bytes to u32")?;

        if recv_status != 200 {
            return Err("Error changing num sm cores".to_string());
        }

        virt_server.num_sm_cores = new_num_sm_cores;
        self.update_virt_server(rpc_id, virt_server);
        Ok(())
    }
    
}

