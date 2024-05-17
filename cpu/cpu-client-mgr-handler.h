#ifndef __CD_CPU_CLIENT_MGR_HANDLER_H__
#define __CD_CPU_CLIENT_MGR_HANDLER_H__

char* init_client_mgr();
void stop_client_mgr();


void change_server(char *server_info); // in cpu-client.c
void resume_connection(void); // in cpu-client.c

#endif // __CD_CPU_CLIENT_MGR_HANDLER_H__