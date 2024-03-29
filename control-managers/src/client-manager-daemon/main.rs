mod resource_manager_handler;
mod vcuda_client_handler;

use std::{os::unix::net::{UnixListener, UnixStream}, sync::{Arc, Mutex}};
use std::net::TcpStream;
use std::io::{self, BufReader, Read, Write};
use std::io::prelude::*;
use std::fmt::{self, Display};
use std::fs::File;
use std::thread;
use toml::Table;



const SOCKET_PATH: &str = "/home/sam/Projects/flyt/control-managers/client-mgr.sock";
const CONFIG_PATH: &str = "/home/sam/Projects/flyt/control-managers/client-mgr.toml";


struct ServerDetails {
    server_address: String,
    rpc_id: u16
}


impl Display for ServerDetails {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.server_address, self.rpc_id)
    }
}

static SERVER_DETAILS: Mutex<ServerDetails> = Mutex::new(ServerDetails {
    server_address: String::new(),
    rpc_id: 0
});

fn is_server_details_set() -> bool {
    let server_details = SERVER_DETAILS.lock().unwrap();
    !server_details.server_address.is_empty()
}

fn update_server_details(server_details: String) -> bool {
    let server_details_split = server_details.split_whitespace().collect::<Vec<&str>>();

    if server_details_split.len() != 2 {
        return false;
    }

    let mut server_details = SERVER_DETAILS.lock().unwrap();
    server_details.server_address = server_details_split[0].to_string();
    server_details.rpc_id = server_details_split[1].parse::<u16>().unwrap();
    true
}




fn resource_manager_connection_handler(mut stream: TcpStream, client_processes: Arc<Mutex<Vec<UnixStream>>>) {
    let mut stream_clone = stream.try_clone().unwrap();
    let mut reader = BufReader::new(&mut stream_clone);

    loop {
        let mut cmd_buf = String::new();
        reader.read_line(&mut cmd_buf).unwrap();

        let cmd = cmd_buf.trim();

        match cmd {
            "PAUSE_CLIENTS" => {
                for mut client in client_processes.lock().unwrap().iter() {
                    writeln!(client, "PAUSE").unwrap();
                }
                
                let mut all_paused = true;

                for client in client_processes.lock().unwrap().iter() {
                    let mut client_reader = BufReader::new(client);
                    let mut buf = String::new();
                    client_reader.read_line(&mut buf).unwrap();
                    
                    match buf.trim() {
                        "200" => {
                            
                        },
                        _ => {
                            all_paused = false;
                            println!("Error pausing client: {}", buf);
                        }
                    }
                }

                if all_paused {
                    writeln!(stream, "200").unwrap();
                } else {
                    writeln!(stream, "500").unwrap();
                }



            },
            _ => {
                println!("Unknown command: {}", cmd);
            }
            
        }
    }
}

fn resource_manager_connection_handler_launch(stream: &mut TcpStream, client_processes: &Arc<Mutex<Vec<UnixStream>>>) {
    let stream_clone = stream.try_clone().unwrap();
    let client_processes_clone = client_processes.clone();

    thread::spawn(move || {
        resource_manager_connection_handler(stream_clone, client_processes_clone);
    });
}

fn connect_to_resource_manager() -> io::Result<TcpStream> {
    let mut file = File::open(CONFIG_PATH).unwrap();
    let mut contents = String::new();

    file.read_to_string(&mut contents).unwrap();

    let config = contents.parse::<Table>().unwrap();

    let resouce_manager_address = [ config["resource_manager"]["address"].as_str().unwrap(), 
                                    config["resource_manager"]["port"].as_str().unwrap() ].join(":");

    println!("Connecting to resource manager at: {}", resouce_manager_address);
    
    TcpStream::connect(resouce_manager_address)
}

fn get_server_details(server_stream: &mut TcpStream) -> io::Result<ServerDetails> {

    let mut request_status = String::new();
    let mut request_payload = String::new();
    let mut reader = BufReader::new(server_stream);
    reader.read_line(&mut request_status).unwrap();
    reader.read_line(&mut request_payload).unwrap();
    if request_status.starts_with("200") {
        let server_details_split = request_payload.trim().split(",").collect::<Vec<&str>>();
        if server_details_split.len() != 2 {
            println!("server details not in correct format: {}", request_payload);
            return io::Result::Err(io::Error::new(io::ErrorKind::Other, "server details not in correct format"));
        }
        io::Result::Ok(ServerDetails {
            server_address: server_details_split[0].to_string(),
            rpc_id: server_details_split[1].parse::<u16>().unwrap()
        })
    } else {
        println!("Error getting server details: {} {}", request_status, request_payload);
        io::Result::Err(io::Error::new(io::ErrorKind::Other, format!("Error getting server details: {} {}", request_status, request_payload)))  
    }
}


fn main() {

    let client_processes  = Arc::new(Mutex::new(Vec::new()));

    let mut resource_manager = connect_to_resource_manager();
    let mut virt_server = match resource_manager {
        io::Result::Ok(ref mut stream) => {
            get_server_details(stream)
        },
        io::Result::Err(_) => {
            resource_manager = io::Result::Err(io::Error::new(io::ErrorKind::Other, "Unable to connect to resource manager"));
            io::Result::Err(io::Error::new(io::ErrorKind::Other, "Unable to connect to resource manager"))
        }
    };

    let listener = UnixListener::bind(SOCKET_PATH).unwrap();

    for stream in listener.incoming() {

        let mut client_stream = stream.unwrap();


        match resource_manager {
            Err(_) => {
                println!("Reconnecting to resource manager");
                resource_manager = connect_to_resource_manager()
            },
            _ => {}
        };

        match resource_manager {
            Err(_) => {
                client_stream.write_all(b"500\nUnable to connect to resource manager").unwrap();
                continue;
            }
            _ => {}
        }

        match virt_server {
            Err(_) => {
                println!("Getting server details again");
                let resource_manager_stream = resource_manager.as_mut().unwrap().try_clone().unwrap();
                virt_server = get_server_details(&resource_manager_stream)
            }
            _ => {}
        };

        match virt_server {
            Err(_) => {
                client_stream.write_all(b"500\nUnable to get server details").unwrap();
                continue;
            }
            _ => {}
        }



        if is_server_details_set() {
            let server_details = SERVER_DETAILS.lock().unwrap();
            client_stream.write_all(format!("200\n{}", server_details).as_bytes()).unwrap();
        } else {
            client_stream.write_all(b"500\nUnable to allocate server process").unwrap();
        }

        client_processes.lock().unwrap().push(client_stream);
        
    }

}