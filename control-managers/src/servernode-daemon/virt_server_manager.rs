use std::{collections::HashMap, env::temp_dir, process::{Child, Command}, sync::{Arc, Mutex}};
use ipc_rs::MessageQueue;


const PROJ_ID: i32 = 0x42;
const VIRT_SERVER_PROGRAM: &str = "cricket-rpc-server";

#[derive(Clone)]
struct VirtServer {
    id: u32,
    gpu_id: u32,
    num_sm_cores: u32,
    gpu_memory: u64,
    process: Arc<Mutex<Child>>,
    message_queue: Arc<Mutex<MessageQueue>>,
}

struct VirtServerManager {
    counter: Mutex<u32>,
    virts_servers: Mutex<HashMap<u32,VirtServer>>,
}



impl VirtServerManager {

    fn new() -> VirtServerManager {
        VirtServerManager {
            counter: Mutex::new(0),
            virts_servers: Mutex::new(HashMap::new()),
        }
    }

    fn add_virt_server(&self, gpu_id: u32, gpu_memory: u64, num_sm_cores: u32, total_gpu_sm_cores: u32) -> Result<u32,String> {
        let rpc_id = {
            let mut counter = self.counter.lock().unwrap();
            *counter += 1;
            *counter
        };
        
        let gpu_cores_percentage = (num_sm_cores as f64 / total_gpu_sm_cores as f64).round() as u32;


        let pipe_path_buf = temp_dir().join( format!("cricket_rpc_server_{}", rpc_id));
        let pipe_path = pipe_path_buf.to_str().ok_or("Error creating pipe path")?;
        
        let key = ipc_rs::PathProjectIdKey::new(pipe_path.to_string(), PROJ_ID);
        let message_queue = MessageQueue::new(ipc_rs::MessageQueueKey::PathKey(key)).create().init().map_err(|_| "Error creating message queue")?;
        
        
        let virt_server_process = Command::new(VIRT_SERVER_PROGRAM)
            .env("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25", gpu_cores_percentage.to_string())
            .arg(rpc_id.to_string())
            .arg(gpu_id.to_string())
            .arg(gpu_memory.to_string())
            .arg(num_sm_cores.to_string())
            .arg(pipe_path)
            .spawn()
            .map_err(|e| format!("Error starting virt server: {}", e))?;


        
        let virt_server = VirtServer {
            id: rpc_id,
            gpu_id: gpu_id,
            num_sm_cores: num_sm_cores,
            gpu_memory: gpu_memory,
            process: Arc::new(Mutex::new(virt_server_process)),
            message_queue: Arc::new(Mutex::new(message_queue)),
        };

        self.virts_servers.lock().unwrap().insert(rpc_id, virt_server);
        
        Ok(rpc_id)
        
    }

    fn remove_virt_server(&self, rpc_id: u32) -> Result<(),String> {
        let mut virt_servers = self.virts_servers.lock().unwrap();
        let virt_server = virt_servers.get(&rpc_id).ok_or("Virt server not found")?;
        
        virt_server.process.lock().unwrap().kill().map_err(|e| format!("Error killing virt server: {}", e))?;
        virt_servers.remove(&rpc_id);
        
        Ok(())
    }

    
    
}

