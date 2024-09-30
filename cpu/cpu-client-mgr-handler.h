#ifndef __CD_CPU_CLIENT_MGR_HANDLER_H__
#define __CD_CPU_CLIENT_MGR_HANDLER_H__

typedef struct server_info {
    int rpc_id;
    int shm_enable;
    int clientd_mqueue_id; // handle to clientd.
    char server_ip[128];
    char shm_backend[64];
} server_info_t;

server_info_t *parse_server_str(char *server_str);

server_info_t *init_client_mgr();
void stop_client_mgr();

void change_server(server_info_t *server_info); // in cpu-client.c
void resume_connection(void); // in cpu-client.c

#endif // __CD_CPU_CLIENT_MGR_HANDLER_H__