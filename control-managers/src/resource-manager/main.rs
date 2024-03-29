mod bookkeeping;
mod servernode_handler;
mod client_handler;

use std::{env, thread};

use servernode_handler::ServerNodesManager;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} client-port servernode-port", args[0]);
        return;
    }

    let client_port = args[1].parse::<u16>().unwrap();
    let servernode_port = args[2].parse::<u16>().unwrap();

    let vm_resource_getter = bookkeeping::VMResourcesGetter::new();

    let server_nodes_manager = ServerNodesManager::new(&vm_resource_getter);
    let client_handler = client_handler::FlytClientManager::new(&server_nodes_manager);

    thread::scope(|s| {
        s.spawn(|| {
            server_nodes_manager.start_servernode_handler(servernode_port);
        });

        s.spawn(|| {
            client_handler.start_flytclient_handler(client_port);
        });
    })

}