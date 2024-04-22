#![allow(dead_code)]

use std::{sync::Arc, thread};

use common::config::SNODE_CONFIG_PATH;
use resource_manager_handler::ResourceManagerHandler;
use virt_server_manager::VirtServerManager;

mod resource_manager_handler;
mod gpu_manager;
mod virt_server_manager;
#[path = "../common/mod.rs"]
mod common;


fn get_mqueue_path() -> String {
    let config = common::utils::Utils::load_config_file(SNODE_CONFIG_PATH);
    config["ipc"]["mqueue-path"].as_str().unwrap().to_string()
}

fn get_virt_server_program_path() -> String {
    let config = common::utils::Utils::load_config_file(SNODE_CONFIG_PATH);
    config["virt-server"]["program-path"].as_str().unwrap().to_string()
}

fn get_resource_mgr_address() -> (String, u16) {
    let config = common::utils::Utils::load_config_file(SNODE_CONFIG_PATH);
    (config["resource-manager"]["address"].as_str().unwrap().to_string(), config["resource-manager"]["port"].as_integer().unwrap() as u16)
}

fn main() {

    env_logger::init();

    let virt_server_manager = Arc::new(VirtServerManager::new(&get_mqueue_path(), get_virt_server_program_path()));
    let resource_manager_handler = ResourceManagerHandler::new(virt_server_manager.clone());
    let (address, port) = get_resource_mgr_address();

    thread::scope(|s| {
        s.spawn(|| {

            loop {
                match resource_manager_handler.connect(&address, port) {
                    Ok(_) => {
                        log::info!("Connected to resource manager");
                        resource_manager_handler.incomming_message_handler();
                    },
                    Err(e) => {
                        log::error!("Error connecting to resource manager: {}", e);
                        log::error!("Retrying in 5 seconds");
                        thread::sleep(std::time::Duration::from_secs(5));
                    }
                }
            }
        });
    });

}