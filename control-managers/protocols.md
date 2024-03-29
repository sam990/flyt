# Control Path Communication Protocols
All control path communication is done using raw streaming sockets {TCP/UNIX}. The protocol is a simple line-based protocol, where each line is a command. The responder will typically send two line responses, the first line being the status of the command, and the second line being the response data. The status is a 3 digit number, where 200 is success, 400 is a client error, and 500 is a server error. The response data is dependent on the command, and may be empty.

## Control Messages

### vCUDA Library -> Client Connection Manager: Connect

#### Request: 
    Empty: vCUDA will open a socket to the Client Connection Manager

#### Response:
    StatusCode
    ServerIP<space>RPCID | ErrorMessage


