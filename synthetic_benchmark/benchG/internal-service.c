#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "civetweb.h"

#include "functions_map.h"

int init_functions();


int handle_request(struct mg_connection *connection) {
    
    char response_body[RESPONSE_MAX_LEN];
    int cont_len;

    const struct mg_request_info *rq_info = mg_get_request_info(connection);

    api_function method_func = get_method_function(rq_info->request_uri);

    if (method_func != NULL) {
        int ret_code = method_func(connection, rq_info->query_string, response_body, RESPONSE_MAX_LEN, &cont_len);
        mg_printf(connection,
                "HTTP/1.0 200 OK\r\n"
                "Content-Type: text/plain\r\n"
                "Content-Length: %d\r\n"
                "\r\n"
                "%s",
                cont_len, response_body);
        
        return 1;
    }

    return 0;
}


int main(int argc, char* argv[]) {
    struct mg_context *ctx;
    struct mg_callbacks callback;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s port num-threads\n", argv[0]);
        exit(1);
    }

    INIT_FUNC_MAP;

    const char* options[] = {
        "listening_ports", argv[1],
        "num_threads", argv[2],
        NULL
    };

    memset(&callback, 0, sizeof(callback));
    
    callback.begin_request = handle_request;

    init_functions();

    ctx = mg_start(&callback, NULL, options);

    printf("Server started. Press enter to stop\n");

    getchar();

    mg_stop(ctx);

    FREE_FUNC_MAP;
    
    return 0;
}