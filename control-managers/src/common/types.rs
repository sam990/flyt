use std::io::{BufReader, Read, Write};



#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MqueueClientControlCommand {
    pub command: [u8; 64],
    pub data: [u8; 64]
}

impl Default for MqueueClientControlCommand {
    fn default() -> Self {
        Self {
            command: [0u8; 64],
            data: [0u8; 64]
        }
    }
}


#[derive(Debug)]
pub struct StreamEnds <T: Read + Write> {
    pub reader: BufReader<T>,
    pub writer: T
} 

impl MqueueClientControlCommand {
    pub fn new(command: &str, data: &str) -> Self {
        let mut command_bytes = [0u8; 64];
        let mut data_bytes = [0u8; 64];

        for (i, byte) in command.as_bytes().iter().enumerate() {
            command_bytes[i] = *byte;
        }

        for (i, byte) in data.as_bytes().iter().enumerate() {
            data_bytes[i] = *byte;
        }

        Self {
            command: command_bytes,
            data: data_bytes
        }
    }

    pub fn as_bytes(&self) -> [u8; 128] {
        let mut bytes = [0u8; 128];
        bytes[..64].copy_from_slice(&self.command);
        bytes[64..].copy_from_slice(&self.data);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; 128]) -> Self {
        let mut command = [0u8; 64];
        let mut data = [0u8; 64];
        command.copy_from_slice(&bytes[..64]);
        data.copy_from_slice(&bytes[64..]);
        Self {
            command: command,
            data: data
        }
    }

    pub fn try_from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 128 {
            return None;
        }
        Some(Self::from_bytes(bytes.try_into().unwrap()))
    }

    fn vec_to_string(vec: &[u8]) -> String {
        let mut v = Vec::<u8>::new();
        for i in vec {
            if *i == 0 {
                break;
            }
            v.push(*i);
        }
        String::from_utf8(v).unwrap()
    }

    pub fn command_str(&self) -> String {
        return Self::vec_to_string(&self.command);
    }

    pub fn data_str(&self) -> String {
        return Self::vec_to_string(&self.data);
    }

}


