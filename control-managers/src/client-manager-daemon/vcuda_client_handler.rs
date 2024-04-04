use std::{io::Write, os::unix::net::UnixStream, sync::RwLock, time::Duration};
use crate::common::api_commands::FlytApiCommand;
use crate::common::utils::Utils;
use crate::common::types::MqueueClientControlCommand;
use crate::resource_manager_handler::VirtServer;

use ipc_rs::{MessageQueue, MessageQueueKey, PathProjectIdKey};

const PROJ_ID: i32 = 0x42;

#[derive(Debug, Clone, Copy)]
struct ClientMessageTypeId {
    send_id: i64,
    recv_id: i64,
}


pub struct VCudaClientManager {
    clients: RwLock<Vec<ClientMessageTypeId>>,
    message_queue: MessageQueue,
    mqueue_path: String
}

impl VCudaClientManager {
    pub fn new(mqueue_path: &str) -> VCudaClientManager {
        let key = PathProjectIdKey::new(mqueue_path.to_string(), PROJ_ID);
        let message_queue = MessageQueue::new(MessageQueueKey::PathKey(key)).create().init().unwrap();
        
        VCudaClientManager {
            clients: RwLock::new(Vec::new()),
            message_queue: message_queue,
            mqueue_path: mqueue_path.to_string()
        }
    }

    fn add_client(&self, client: ClientMessageTypeId) {
        self.clients.write().unwrap().push(client);
    }

    pub fn num_active_clients(&self) -> usize {
        self.clients.read().unwrap().len()
    }

    pub fn pause_clients(&self) -> u32 {
        
        let mut pause_count = 0u32;

        let bytes = MqueueClientControlCommand::new(FlytApiCommand::CLIENTD_VCUDA_PAUSE, "").as_bytes();

        for client in self.clients.read().unwrap().iter() {
            
            if self.message_queue.send(&bytes, client.send_id).is_err() {
                println!("Error sending pause command to client: {}", client.send_id);
                continue;
            }

            let recv_bytes = self.message_queue.recv_type_timed(client.recv_id, Duration::from_secs(2));
            if recv_bytes.is_err() {
                println!("Error receiving response from client: {}", client.send_id);
                continue;
            }
            let recv_bytes = recv_bytes.unwrap();
            let response = Utils::convert_bytes_to_u32(&recv_bytes);
            if response.is_none() {
                println!("Error parsing response from client: {}", client.send_id);
                continue;
            }
            
            if response.unwrap() == 200 {
                pause_count += 1;
            }
            else {
                println!("Error pausing client: {}", client.send_id);
            }

        }

        pause_count
    }

    pub fn change_virt_server(&self, virt_server: &VirtServer) -> u32 {
        
        let mut change_count = 0u32;

        let address_str = format!("{},{}", virt_server.address, virt_server.rpc_id);

        let bytes = MqueueClientControlCommand::new(FlytApiCommand::CLIENTD_VCUDA_CHANGE_VIRT_SERVER, &address_str).as_bytes();
        
        for client in self.clients.read().unwrap().iter() {
            if self.message_queue.send(&bytes, client.send_id).is_err() {
                println!("Error sending change virt server command to client: {}", client.send_id);
                continue;
            }

            let recv_bytes = self.message_queue.recv_type_timed(client.recv_id, Duration::from_secs(2));
            if recv_bytes.is_err() {
                println!("Error receiving response from client: {}", client.send_id);
                continue;
            }
            let recv_bytes = recv_bytes.unwrap();

            let response = Utils::convert_bytes_to_u32(&recv_bytes);
            if response.is_none() {
                println!("Error parsing response from client: {}", client.send_id);
                continue;
            }
            
            if response.unwrap() == 200 {
                change_count += 1;
            }
            else {
                println!("Error changing virt server for client: {}", client.send_id);
            }
        }

        change_count
    }


    pub fn resume_clients(&self) -> u32 {
        
        let mut resume_count = 0u32;

        let bytes = MqueueClientControlCommand::new(FlytApiCommand::CLIENTD_VCUDA_RESUME, "").as_bytes();

        for client in self.clients.read().unwrap().iter() {
            if self.message_queue.send(&bytes, client.send_id).is_err() {
                println!("Error sending resume command to client: {}", client.send_id);
                continue;
            }

            let recv_bytes = self.message_queue.recv_type_timed(client.recv_id, Duration::from_secs(2));
            if recv_bytes.is_err() {
                println!("Error receiving response from client: {}", client.send_id);
                continue;
            }
            let recv_bytes = recv_bytes.unwrap();
            
            let response = Utils::convert_bytes_to_u32(&recv_bytes);
            if response.is_none() {
                println!("Error parsing response from client: {}", client.send_id);
                continue;
            }
            
            if response.unwrap() == 200 {
                resume_count += 1;
            }
            else {
                println!("Error resuming client: {}", client.send_id);
            }
        }

        resume_count
    }

    pub fn send_virt_server(&self, mut stream: UnixStream, virt_server: &VirtServer) {
        stream.write_all(format!("200\n{},{}\n", virt_server.address, virt_server.rpc_id).as_bytes()).unwrap();
    }

    pub fn remove_closed_clients<F>(&self, notify_fn: F) 
    where F: Fn() -> bool {

        self.clients.write().unwrap().retain(|client| {
            Utils::is_ping_active(&self.message_queue, client.send_id, client.recv_id)
        });

        if self.clients.write().unwrap().len() == 0 {
            println!("All clients disconnected. Exiting...");
            notify_fn();
        }
    }

    pub fn listen_to_clients(&self) {
        let key = PathProjectIdKey::new(self.mqueue_path.clone(), PROJ_ID);
        let message_queue = MessageQueue::new(MessageQueueKey::PathKey(key)).create().init().unwrap();

        loop {
            let recv_bytes = message_queue.recv_type(1).unwrap();
            let client_pid = Utils::convert_bytes_to_u32(&recv_bytes);
            if client_pid.is_none() {
                println!("Error converting bytes to u32");
                continue;
            }
            let client_pid = client_pid.unwrap();
            
            let recive_id = ((client_pid as u64) << 32 ) | client_pid as u64;

            let client_msg_id = ClientMessageTypeId {
                send_id: client_pid as i64,
                recv_id: recive_id as i64
            };

            self.add_client(client_msg_id);

        }
    }


}