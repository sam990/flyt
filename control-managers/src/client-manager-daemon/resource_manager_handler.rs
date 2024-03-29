use std::{net::TcpStream, thread};


pub struct ResourceManagerHandler {
    server_ip: String,
    server_port: u16,
    stream: Option<TcpStream>,
    command_reader_thread: Option<thread::JoinHandle<()>>
}


impl ResourceManagerHandler {
    pub fn new(server_ip: String, server_port: u16) -> ResourceManagerHandler {

        ResourceManagerHandler {
            server_ip: server_ip,
            server_port: server_port,
            stream: None,
            command_reader_thread: None
        }
    }

    pub fn try_connected(&mut self) -> bool {
        if self.stream.is_some() {
            return true;
        }
        let stream = TcpStream::connect(format!("{}:{}", self.server_ip, self.server_port));
        match stream {
            Ok(stream) => {
                self.stream = Some(stream);
                true
            }
            Err(error) => {
                println!("Error connecting to server: {}", error);
                false
            }
        }
    }


    fn launch_cmd_reader_thread(&mut self) {
        let stream_clone = self.stream.unwrap().try_clone().unwrap();
        let command_reader_thread = thread::spawn(move || {
            
        });
        self.command_reader_thread = Some(command_reader_thread);
    }

}