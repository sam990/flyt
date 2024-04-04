#![allow(dead_code)]

mod resource_manager_handler;
mod vcuda_client_handler;
#[path = "../common/mod.rs"]
mod common;

use std::fs::File;
use std::io::Read;
use std::thread;
use toml::Table;
use vcuda_client_handler::VCudaClientManager;

use crate::resource_manager_handler::ResourceManagerHandler;

const CONFIG_PATH: &str = "/home/sam/Projects/flyt/control-managers/client-mgr.toml";

fn get_mqueue_path() -> String {
    let mut file = File::open(CONFIG_PATH).unwrap();
    let mut contents = String::new();

    file.read_to_string(&mut contents).unwrap();

    let config = contents.parse::<Table>().unwrap();

    config["vcuda-client"]["mqueue-path"].as_str().unwrap().to_string()
}

fn get_vcuda_process_monitor_period() -> u64 {
    let mut file = File::open(CONFIG_PATH).unwrap();
    let mut contents = String::new();

    file.read_to_string(&mut contents).unwrap();

    let config = contents.parse::<Table>().unwrap();

    let period = || -> Option<u64> {
        Some(config.get("vcuda-client")?.get("process_monitor_period")?.as_integer()? as u64)
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

    (config["resource-manager"]["address"].as_str().unwrap().to_string(), config["resource-manager"]["port"].as_integer().unwrap() as u16)
}


fn main() {

    let (resource_manager_address, resource_manager_port) = get_resource_mgr_address();
    let mqueue_path = get_mqueue_path();

    let client_mgr = VCudaClientManager::new(&mqueue_path);
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
            client_mgr.listen_to_clients();
        });
    });

    

}