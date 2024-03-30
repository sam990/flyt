mod resource_manager_handler;
mod vcuda_client_handler;
#[path = "../common/mod.rs"]
mod common;

use std::{fs::File, os::unix::net::UnixListener};
use std::io::{Read, Write};
use std::thread;
use toml::Table;
use vcuda_client_handler::VCudaClientManager;

use crate::resource_manager_handler::ResourceManagerHandler;

const SOCKET_PATH: &str = "/home/sam/Projects/flyt/control-managers/client-mgr.sock";
const CONFIG_PATH: &str = "/home/sam/Projects/flyt/control-managers/client-mgr.toml";

fn get_vcuda_process_monitor_period() -> u64 {
    let mut file = File::open(CONFIG_PATH).unwrap();
    let mut contents = String::new();

    file.read_to_string(&mut contents).unwrap();

    let config = contents.parse::<Table>().unwrap();

    let period = || -> Option<u64> {
        Some(config.get("vcuda_client")?.get("process_monitor_period")?.as_integer()? as u64)
    };

    match period() {
        Some(p) => p as u64,
        None => 60u64
    }
}

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

    let client_mgr = VCudaClientManager::new();
    let res_mgr = ResourceManagerHandler::new(resource_manager_address, resource_manager_port, &client_mgr);

    thread::scope(|s| {
        s.spawn(|| {
            let period = get_vcuda_process_monitor_period();
            loop {
                thread::sleep(std::time::Duration::from_secs(period));
                client_mgr.remove_closed_clients(|| res_mgr.notify_zero_clients() );
            }
        });

        s.spawn(|| {
            incoming_vcuda_client_handler(&res_mgr, &client_mgr, s);
        });
    });

    

}