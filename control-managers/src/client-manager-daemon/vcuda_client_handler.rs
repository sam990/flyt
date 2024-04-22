use std::{sync::RwLock, time::Duration};
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

        if std::path::Path::new(mqueue_path).exists() == false {
            std::fs::File::create(mqueue_path).unwrap();
        }

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

        log::info!("Pausing clients...");
        
        let mut pause_count = 0u32;

        let bytes = MqueueClientControlCommand::new(FlytApiCommand::CLIENTD_VCUDA_PAUSE, "").as_bytes();

        for client in self.clients.read().unwrap().iter() {
            
            if self.message_queue.send(&bytes, client.send_id).is_err() {
                log::error!("Error sending pause command to client: {}", client.send_id);
                continue;
            }

            let recv_bytes = self.message_queue.recv_type_timed(client.recv_id, Duration::from_secs(2));
            if recv_bytes.is_err() {
                log::error!("Error receiving response from client: {}", client.send_id);
                continue;
            }
            let recv_bytes = recv_bytes.unwrap();
            let response = Utils::convert_bytes_to_u32(&recv_bytes);
            if response.is_none() {
                log::error!("Error parsing response from client: {}", client.send_id);
                continue;
            }
            
            if response.unwrap() == 200 {
                pause_count += 1;
            }
            else {
                log::error!("Error pausing client: {}", client.send_id);
            }

        }

        log::info!("Paused #clients: {}", pause_count);

        pause_count
    }

    pub fn change_virt_server(&self, virt_server: &VirtServer) -> u32 {
        
        log::info!("Changing virt server for clients...");

        let mut change_count = 0u32;

        let address_str = format!("{},{}", virt_server.address, virt_server.rpc_id);

        let bytes = MqueueClientControlCommand::new(FlytApiCommand::CLIENTD_VCUDA_CHANGE_VIRT_SERVER, &address_str).as_bytes();
        
        for client in self.clients.read().unwrap().iter() {
            if self.message_queue.send(&bytes, client.send_id).is_err() {
                log::error!("Error sending change virt server command to client: {}", client.send_id);
                continue;
            }

            let recv_bytes = self.message_queue.recv_type_timed(client.recv_id, Duration::from_secs(2));
            if recv_bytes.is_err() {
                log::error!("Error receiving response from client: {}", client.send_id);
                continue;
            }
            let recv_bytes = recv_bytes.unwrap();

            let response = Utils::convert_bytes_to_u32(&recv_bytes);
            if response.is_none() {
                log::error!("Error parsing response from client: {}", client.send_id);
                continue;
            }
            
            if response.unwrap() == 200 {
                change_count += 1;
            }
            else {
                log::error!("Error changing virt server for client: {}", client.send_id);
            }
        }

        log::info!("Changed virt server for #clients: {}", change_count);

        change_count
    }


    pub fn resume_clients(&self) -> u32 {

        log::info!("Resuming clients...");
        
        let mut resume_count = 0u32;

        let bytes = MqueueClientControlCommand::new(FlytApiCommand::CLIENTD_VCUDA_RESUME, "").as_bytes();

        for client in self.clients.read().unwrap().iter() {
            if self.message_queue.send(&bytes, client.send_id).is_err() {
                log::error!("Error sending resume command to client: {}", client.send_id);
                continue;
            }

            let recv_bytes = self.message_queue.recv_type_timed(client.recv_id, Duration::from_secs(2));
            if recv_bytes.is_err() {
                log::error!("Error receiving response from client: {}", client.send_id);
                continue;
            }
            let recv_bytes = recv_bytes.unwrap();
            
            let response = Utils::convert_bytes_to_u32(&recv_bytes);
            if response.is_none() {
                log::error!("Error parsing response from client: {}", client.send_id);
                continue;
            }
            
            if response.unwrap() == 200 {
                resume_count += 1;
            }
            else {
                log::error!("Error resuming client: {}", client.send_id);
            }
        }

        log::info!("Resumed #clients: {}", resume_count);

        resume_count
    }

    pub fn remove_closed_clients<F>(&self, notify_fn: F) 
    where F: Fn() -> bool {

        self.clients.write().unwrap().retain(|client| {
            Utils::is_ping_active(&self.message_queue, client.send_id, client.recv_id)
        });

        if self.clients.write().unwrap().len() == 0 {
            log::info!("No client connected... notifying...");
            notify_fn();
        }
    }

    pub fn listen_to_clients<F>(&self, virt_server_getter: F) where F: Fn() -> Option<VirtServer> {
        let key = PathProjectIdKey::new(self.mqueue_path.clone(), PROJ_ID);
        let message_queue = MessageQueue::new(MessageQueueKey::PathKey(key)).create().init().unwrap();
        log::info!("Flyt client manager listening to clients...");
        loop {
            let recv_bytes = message_queue.recv_type(1);
            if recv_bytes.is_err() {
                log::error!("Error receiving message from client");
                continue;
            }
            let recv_bytes = recv_bytes.unwrap();
            let client_pid = Utils::convert_bytes_to_u32(&recv_bytes);
            if client_pid.is_none() {
                log::error!("Error converting bytes to u32");
                continue;
            }
            let client_pid = client_pid.unwrap();
            
            let receive_id = (client_pid as u64) << 32;

            let client_msg_id = ClientMessageTypeId {
                send_id: client_pid as i64,
                recv_id: receive_id as i64
            };

            log::info!("Client connected: {}", client_pid);

            let virt_server = virt_server_getter();
            if virt_server.is_none() {
                log::error!("Error getting virt server details");
                
                let bytes = MqueueClientControlCommand::new("500", "").as_bytes();
                message_queue.send(&bytes, client_msg_id.send_id).unwrap();
                continue;
            }

            let virt_server = virt_server.unwrap();
            let address_str = format!("{},{}", virt_server.address, virt_server.rpc_id);
            let bytes = MqueueClientControlCommand::new("200", &address_str).as_bytes();
            match message_queue.send(&bytes, client_msg_id.send_id) {
                Ok(_) => {
                    log::info!("Sent virt server details to client: {}", client_pid);
                    self.add_client(client_msg_id);
                },
                Err(_) => {
                    log::error!("Error sending virt server details to client: {}", client_pid);
                }
            }

        }
    }


}