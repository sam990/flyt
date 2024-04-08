#![allow(dead_code)]

use std::{
    io::{BufRead, Write},
    os::unix::net::UnixStream,
};

use clap::{Parser, Subcommand};
use comfy_table::Table;
use common::api_commands::FrontEndCommand;

#[path = "../common/mod.rs"]
mod common;

const RESOURCE_MANAGER_CONFIG: &str =
    "/home/sam/Projects/flyt/control-managers/resource-mgr-config.toml";

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    ListVms,
    ListServernodes,
    ListVirtServers,
    ChangeConfig {
        #[arg(short, long, help = "IP address of the VM to change resources for")]
        ip: String,
        #[clap(flatten)]
        new_resources: NewResourcesOption,
    },
}
#[derive(Debug, clap::Args, Clone)]
#[group(required = true)]
struct NewResourcesOption {
    #[arg(short, long, help = "Number of SM cores to allocate")]
    sm_cores: Option<u32>,
    #[arg(short, long, help = "Amount of memory to allocate")]
    memory: Option<u64>,
}

fn get_stream_path() -> String {
    let config = common::utils::Utils::load_config_file(RESOURCE_MANAGER_CONFIG);
    config["ipc"]["frontend-socket"]
        .as_str()
        .unwrap()
        .to_string()
}

fn main() {
    let args = Args::parse();
    let stream_path = get_stream_path();

    let stream = UnixStream::connect(stream_path).unwrap();

    match args.cmd {
        Commands::ListVms => {
            list_vms(stream);
        }
        Commands::ListServernodes => {
            list_servernodes(stream);
        }
        Commands::ListVirtServers => list_virt_servers(stream),
        Commands::ChangeConfig { ip, new_resources } => {
            change_resources(stream, ip, new_resources);
        }
    }
}

fn list_vms(mut stream: UnixStream) {
    stream
        .write_all(FrontEndCommand::LIST_VMS.as_bytes())
        .unwrap();
    let mut buffer = String::new();
    let mut reader = std::io::BufReader::new(stream);

    reader.read_line(&mut buffer).unwrap();

    if buffer.trim() != "200" {
        println!("Error: {}", buffer);
        reader.read_line(&mut buffer).unwrap();
        println!("{}", buffer);
        return;
    }

    buffer.clear();
    reader.read_line(&mut buffer).unwrap();
    let num_vms = buffer.trim().parse::<usize>().unwrap();

    let mut table = Table::new();

    table.set_header(vec![
        "VM IP",
        "VirtServer IP",
        "VirtServer RPC ID",
        "SM Cores",
        "Memory",
        "Is Active",
    ]);

    for _ in 0..num_vms {
        buffer.clear();
        reader.read_line(&mut buffer).unwrap();
        let fields = buffer.trim().split(',').collect::<Vec<&str>>();
        table.add_row(fields);
    }

    println!("{table}");
}

fn list_servernodes(mut stream: UnixStream) {
    stream
        .write_all(FrontEndCommand::LIST_SERVER_NODES.as_bytes())
        .unwrap();
    let mut buffer = String::new();
    let mut reader = std::io::BufReader::new(stream);

    reader.read_line(&mut buffer).unwrap();

    if buffer.trim() != "200" {
        println!("Error: {}", buffer);
        reader.read_line(&mut buffer).unwrap();
        println!("{}", buffer);
        return;
    }

    buffer.clear();
    reader.read_line(&mut buffer).unwrap();
    let num_servernodes = buffer.trim().parse::<usize>().unwrap();

    let mut response = String::new();

    // format: ipaddr,num_gpus
    for _ in 0..num_servernodes {
        let fields = buffer.trim().split(',').collect::<Vec<&str>>();
        let ipaddr = fields[0];
        let num_gpus = fields[1].parse::<usize>().unwrap();

        response.push_str(format!("ServerNode IP: {}\n", ipaddr).as_str());
        let mut table = Table::new();

        // format: gpuid,name,memory,allocated_memory,compute_units,allocated_compute_units
        table.set_header(vec![
            "GPU ID",
            "GPU Name",
            "GPU Memory",
            "Allocated GPU Memory",
            "GPU Compute Units",
            "Allocated GPU Compute Units",
        ]);

        for _ in 0..num_gpus {
            buffer.clear();
            reader.read_line(&mut buffer).unwrap();
            let fields = buffer.trim().split(',').collect::<Vec<&str>>();
            table.add_row(fields);
        }

        response.push_str(&format!("{table}\n"));
    }

    println!("{}", response);
}

fn list_virt_servers(mut stream: UnixStream) {
    stream
        .write_all(FrontEndCommand::LIST_VIRT_SERVERS.as_bytes())
        .unwrap();
    let mut buffer = String::new();
    let mut reader = std::io::BufReader::new(stream);

    reader.read_line(&mut buffer).unwrap();

    if buffer.trim() != "200" {
        println!("Error: {}", buffer);
        reader.read_line(&mut buffer).unwrap();
        println!("{}", buffer);
        return;
    }

    buffer.clear();
    reader.read_line(&mut buffer).unwrap();
    let num_virt_servers = buffer.trim().parse::<usize>().unwrap();

    // format: ipaddr,rpc_id,gpu_id,compute_units,memory
    let mut table = Table::new();
    table.set_header(vec![
        "VirtServer IP",
        "VirtServer RPC ID",
        "GPU ID",
        "Compute Units",
        "Memory",
    ]);

    for _ in 0..num_virt_servers {
        buffer.clear();
        reader.read_line(&mut buffer).unwrap();
        let fields = buffer.trim().split(',').collect::<Vec<&str>>();
        table.add_row(fields);
    }

    println!("{}", table);
}

fn change_resources(mut stream: UnixStream, vm_ip: String, new_resources: NewResourcesOption) {
    let command = if new_resources.sm_cores.is_some() && new_resources.memory.is_some() {
        format!(
            "{}\n{},{},{}\n",
            FrontEndCommand::CHANGE_SM_CORES_AND_MEMORY,
            vm_ip,
            new_resources.sm_cores.unwrap(),
            new_resources.memory.unwrap()
        )
    } else if let Some(sm_cores) = new_resources.sm_cores {
        format!(
            "{}\n{},{}\n",
            FrontEndCommand::CHANGE_SM_CORES,
            vm_ip,
            sm_cores
        )
    } else if let Some(memory) = new_resources.memory {
        format!("{}\n{},{}\n", FrontEndCommand::CHANGE_MEMORY, vm_ip, memory)
    } else {
        "".to_string()
    };

    if command.is_empty() {
        println!("Invalid command");
        return;
    }

    stream.write_all(command.as_bytes()).unwrap();

    let response = common::utils::Utils::read_response(&mut stream, 2);

    println!("{}: {}", response[0], response[1]);
}
