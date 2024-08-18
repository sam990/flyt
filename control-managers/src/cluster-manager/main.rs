#![allow(dead_code)]

mod bookkeeping;
mod servernode_handler;
mod client_handler;
mod cli_frontend;
mod frontend_handler;
#[path = "../common/mod.rs"]
mod common;

use std::thread;

use frontend_handler::FrontendHandler;
use servernode_handler::ServerNodesManager;

fn main() {

    env_logger::init();

    let ( servernode_port, client_port) = bookkeeping::get_ports();

    let vm_resource_getter = bookkeeping::VMResourcesGetter::new(); 

    let server_nodes_manager = ServerNodesManager::new(&vm_resource_getter);
    let client_handler = client_handler::FlytClientManager::new(&server_nodes_manager);
    let frontend_handler = FrontendHandler::new(&client_handler, &server_nodes_manager);

    thread::scope(|s| {
        s.spawn(|| {
            server_nodes_manager.start_servernode_handler(servernode_port);
        });

        s.spawn(|| {
            client_handler.start_flytclient_handler(client_port, s);
        });

        s.spawn(|| {
            frontend_handler.start_listening(crate::cli_frontend::get_stream_path().as_str());
        });
    });

}