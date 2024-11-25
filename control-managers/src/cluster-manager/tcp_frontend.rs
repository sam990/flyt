#![allow(dead_code)]

use std::{io::{Read, Write}, net::TcpListener, os::unix::net::UnixStream, time::SystemTime};

use cli_frontend::{change_resources, get_stream_path, increase_resources, migrate_vm_auto, NewResourcesOption, set_vm_grouping};

mod cli_frontend;
#[path = "../common/mod.rs"]
mod common;

const INCREASE_SM_DELTA: u32 = 8;

fn get_stream() -> UnixStream {
    let stream_path = get_stream_path();
    UnixStream::connect(stream_path).unwrap()
}

fn main() {
    let mut curr_sm_cores = 10;
    let mut migration_done = false;
    let migration_vm_mem = 8192 * 1024 * 1024;

    let listener = TcpListener::bind("0.0.0.0:32578").unwrap();
    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        println!("Connection established!");
        let vm_ip = stream.peer_addr().unwrap().ip().to_string();

        let mut buf = [0; 2];

        stream.read_exact(&mut buf).unwrap();
        let num = buf[1] as u8;

        println!("command: {} args: {} ",buf[0], num);
        /*
        if num == None {
            buf[1] = 0;
        }
        else {
            buf[1] = num.to_digit(10).unwrap() as u8;
        }
        */
        buf[1] = num;


        match buf[0] {
            1u8 => {
                println!("Received inc_cores command");             

                let mgr_stream = get_stream();

                //TODO: take client id. For now hardcoding to first client id
                let res = increase_resources(mgr_stream, &vm_ip, buf[1].into(), NewResourcesOption{ sm_cores: Some(INCREASE_SM_DELTA), memory: None }, false);

                if res.success {
                    stream.write(&[0u8]).unwrap();
                } else {
                    stream.write(&[1u8]).unwrap();
                }

                let mut file = match std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open("resource_tracker.log")
                {
                    Ok(file) => file,
                    Err(e) => {
                        log::error!("Error opening log file: {}", e);
                        return;
                    }
                };

                let current_time = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_micros();
                if let Err(e) = writeln!(file, "{}: {},{},{},{},{},{},{}", current_time, vm_ip, buf[1], res.success, res.sm_cores, res.memory, res.time_taken.as_micros(), res.error) {
                    log::error!("Error writing to log file: {}", e);
                }

            }
            
            2u8 => {
                println!("Received inc_cores_with_migration command client id: {}", buf[1]);
                /*
                if migration_done && curr_sm_cores >= 64 {
                    println!("Cannot increment cores. Already at max capacity.");
                    stream.write(&[1u8]).unwrap();
                } else if !migration_done && curr_sm_cores >= 48 {
                    */
                    let delta = 64 - curr_sm_cores;
                    curr_sm_cores = 54;
                    let mgr_stream = get_stream();

                    // migrate_vm(mgr_stream, vm_ip, migration_vm_ip.to_string(), 0, sm_cores, memory)
                    // TODO: client-id hardcoded to 0`
                    migrate_vm_auto(mgr_stream.try_clone().unwrap(), vm_ip.clone(), buf[1].into(), curr_sm_cores, migration_vm_mem);
                    let res = increase_resources(mgr_stream, &vm_ip, buf[1].into(), NewResourcesOption{ sm_cores: Some(delta), memory: None }, true);
                    //migration_done = true;
                    stream.write(&[0u8]).unwrap();
                    /*
                }
                else {
                    println!("Incrementing cores...");
                    curr_sm_cores += 16;
                    let mgr_stream = get_stream();

                    //TODO: client-id hardcoded to 0
                    change_resources(mgr_stream, &vm_ip, buf[1].into(), NewResourcesOption{ sm_cores: Some(curr_sm_cores), memory: None });

                    stream.write(&[0u8]).unwrap();
                }
                */
            }
            3u8 => {
                println!("Received dec_cores command");             

                let mgr_stream = get_stream();

                //TODO: take client id. For now hardcoding to first client id
                let res = increase_resources(mgr_stream, &vm_ip, buf[1].into(), NewResourcesOption{ sm_cores: Some(INCREASE_SM_DELTA), memory: None }, true);

                if res.success {
                    stream.write(&[0u8]).unwrap();
                } else {
                    stream.write(&[1u8]).unwrap();
                }

                let mut file = match std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open("resource_tracker.log")
                {
                    Ok(file) => file,
                    Err(e) => {
                        log::error!("Error opening log file: {}", e);
                        return;
                    }
                };

                let current_time = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_micros();
                if let Err(e) = writeln!(file, "{}: {},{},{},{},{},{},{}", current_time, vm_ip, buf[1], res.success, res.sm_cores, res.memory, res.time_taken.as_micros(), res.error) {
                    log::error!("Error writing to log file: {}", e);
                }

            }
            4u8 => {
                println!("Set VM Grouping {}", buf[1]);             

                let mgr_stream = get_stream();

                //TODO: take client id. For now hardcoding to first client id
                let res = set_vm_grouping(mgr_stream, &vm_ip, buf[1].into());

                stream.write(&[0u8]).unwrap();

            }

            _ => {
                println!("Received unknown command: {}", buf[0]);
            }
        }

    }

}
