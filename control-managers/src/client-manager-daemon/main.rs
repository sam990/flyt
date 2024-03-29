mod resource_manager_handler;
mod vcuda_client_handler;
#[path = "../common/mod.rs"]
mod common;

use std::{os::unix::net::{UnixListener, UnixStream}, sync::{Arc, Mutex}};
use std::net::TcpStream;
use std::io::{self, BufReader, Read, Write};
use std::io::prelude::*;
use std::fmt::{self, Display};
use std::fs::File;
use std::thread;
use toml::Table;
use vcuda_client_handler::VCudaClientManager;

use crate::resource_manager_handler::{ ResourceManagerHandler, VirtServer };

const SOCKET_PATH: &str = "/home/sam/Projects/flyt/control-managers/client-mgr.sock";
const CONFIG_PATH: &str = "/home/sam/Projects/flyt/control-managers/client-mgr.toml";


fn get_resource_mgr_address() -> (String, u16) {
    let mut file = File::open(CONFIG_PATH).unwrap();
    let mut contents = String::new();

    file.read_to_string(&mut contents).unwrap();

    let config = contents.parse::<Table>().unwrap();

    (config["resource_manager"]["address"].as_str().unwrap().to_string(), config["resource_manager"]["port"].as_integer().unwrap() as u16)
}

fn incoming_vcuda_client_handler<'a>(res_mgr: &'a ResourceManagerHandler, client_mgr: &'a VCudaClientManager, scope: &'a thread::Scope<'a,'_>) {
    let listener = UnixListener::bind(SOCKET_PATH).unwrap();

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let vserver = &mut res_mgr.get_virt_server(scope);
                match vserver {
                    Some(vserver) => {
                        client_mgr.send_virt_server(stream.try_clone().unwrap(), &vserver);
                        client_mgr.add_client(stream)
                    },
                    None => {
                        stream.write_all("500\nUnable to allocate virt server\n".as_bytes()).unwrap();
                    }
                }
            },
            Err(error) => {
                println!("Error accepting connection: {}", error);
            }
        }
        
    }
}


fn main() {

    let (resource_manager_address, resource_manager_port) = get_resource_mgr_address();

    let mut res_mgr = ResourceManagerHandler::new(resource_manager_address, resource_manager_port);
    let client_mgr = VCudaClientManager::new();


    thread::scope(|s| {
        s.spawn(|| {
            //
        });

        s.spawn(|| {
            incoming_vcuda_client_handler(&mut res_mgr, &client_mgr, s);
        });
    });

    

}