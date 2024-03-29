use std::{io::{Read, Write}, os::unix::net::UnixStream};




pub struct VCudaClientManager {
    clients: Vec<UnixStream>
}

impl VCudaClientManager {
    pub fn new() -> VCudaClientManager {
        VCudaClientManager {
            clients: Vec::new()
        }
    }

    pub fn add_client(&mut self, client: UnixStream) {
        self.clients.push(client);
    }

    pub fn remove_client(&mut self, client: UnixStream) {
        self.clients.retain(|c| c != &client);
    }

    pub fn get_clients(&self) -> &Vec<UnixStream> {
        &self.clients
    }

    pub fn num_active_clients(&self) -> usize {
        self.clients.len()
    }

    pub fn pause_clients (&self) -> bool {
        
        let mut success = true;

        for mut client in &self.clients {
            let client_clone = client.try_clone().unwrap();
            client.write_all(b"PAUSE\n" ).unwrap();
        }

        false
    }
}