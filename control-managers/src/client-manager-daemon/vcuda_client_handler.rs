use std::{io::Write, os::unix::net::UnixStream, sync::RwLock};
use crate::common::api_commands::FlytApiCommand;
use crate::common::utils::Utils;
use crate::resource_manager_handler::VirtServer;

pub struct VCudaClientManager {
    clients: RwLock<Vec<UnixStream>>
}

impl VCudaClientManager {
    pub fn new() -> VCudaClientManager {
        VCudaClientManager {
            clients: RwLock::new(Vec::new())
        }
    }

    pub fn add_client(&self, client: UnixStream) {
        self.clients.write().unwrap().push(client);
    }

    pub fn num_active_clients(&self) -> usize {
        self.clients.read().unwrap().len()
    }

    pub fn pause_clients(&self) -> u32 {
        
        let mut change_count = 0u32;

        for mut client in self.clients.read().unwrap().iter() {
            let client_clone = client.try_clone().unwrap();
            client.write_all(format!("{}\n", FlytApiCommand::CLIENTD_VCUDA_PAUSE).as_bytes()).unwrap();
            let response = Utils::read_response(client_clone, 1);
            if response[0] == "200" {
                change_count += 1;
            }
        }

        change_count
    }

    pub fn change_virt_server(&self, virt_server: &VirtServer) -> u32 {
        
        let mut change_count = 0u32;

        for mut client in self.clients.read().unwrap().iter() {
            let client_clone = client.try_clone().unwrap();
            client.write_all(format!("{}\n{},{}\n", FlytApiCommand::CLIENTD_VCUDA_CHANGE_VIRT_SERVER, virt_server.address, virt_server.rpc_id).as_bytes()).unwrap();
            let response = Utils::read_response(client_clone, 1);
            if response[0] == "200" {
                change_count += 1;
            }
        }

        change_count
    }


    pub fn resume_clients(&self) -> u32 {
        
        let mut change_count = 0u32;

        for mut client in self.clients.read().unwrap().iter() {
            let client_clone = client.try_clone().unwrap();
            client.write_all(format!("{}\n", FlytApiCommand::CLIENTD_VCUDA_RESUME).as_bytes()).unwrap();
            let response = Utils::read_response(client_clone, 1);
            if response[0] == "200" {
                change_count += 1;
            }
        }

        change_count
    }

    pub fn send_virt_server(&self, mut stream: UnixStream, virt_server: &VirtServer) {
        stream.write_all(format!("200\n{},{}\n", virt_server.address, virt_server.rpc_id).as_bytes()).unwrap();
    }

    pub fn remove_closed_clients<F: Fn() -> bool>(&self, notify_fn: F) {
        self.clients.write().unwrap().retain(|c| {
            let client_clone = c.try_clone().unwrap();
            Utils::is_stream_alive(client_clone)
        });

        if self.clients.write().unwrap().len() == 0 {
            println!("All clients disconnected. Exiting...");
            notify_fn();
        }
    }


}