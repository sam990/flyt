// Copyright (c) 2024-2026 SynerG Lab, IITB

#![allow(dead_code)]

mod bookkeeping;
mod servernode_handler;
mod client_handler;
mod cli_frontend;
mod frontend_handler;
mod metrics_handler;
#[path = "../common/mod.rs"]
mod common;

use std::thread;

use frontend_handler::FrontendHandler;
use servernode_handler::ServerNodesManager;
use metrics_handler::MetricsHandler;

fn main() {

    env_logger::init();

    let ( servernode_port, client_port) = bookkeeping::get_ports();
    let ( metrics_port, metrics_interval) = bookkeeping::get_metrics_port();

    let vm_resource_getter = bookkeeping::VMResourcesGetter::new();

    let server_nodes_manager = ServerNodesManager::new(&vm_resource_getter);
    let client_handler = client_handler::FlytClientManager::new(&server_nodes_manager);
    let frontend_handler = FrontendHandler::new(&client_handler, &server_nodes_manager);
    let metrics_handler = MetricsHandler::new(&client_handler, &server_nodes_manager);

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
        /*
        s.spawn(|| {
            metrics_handler.start_metrics_handler(metrics_port);
        });
        s.spawn(|| {
            metrics_handler.start_period_handler(metrics_interval as u64);
        });
        */
    });

}
