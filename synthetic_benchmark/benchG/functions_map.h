
#ifndef __FUNCTIONS_MAP_H__
#define __FUNCTIONS_MAP_H__

#include "civetweb.h"

#define RESPONSE_MAX_LEN 256
#define FILE_LINE_BUF_LEN 256
#define HASH_TABLE_CAPACITY 21

typedef int (*api_function)(struct mg_connection *, const char*, char* , int , int* );

typedef struct {
    const char* method_uri;
    api_function method_function;
} method_map;


void init_hash_table(int capacity);
int is_table_initialized(void);
void free_hash_table(void);
int insert_item(const char* method_uri, api_function method_func_ptr);

api_function get_method_function(const char *);

void init_function_maps(void);


#define INIT_FUNC_MAP \
    do { \
        init_hash_table(HASH_TABLE_CAPACITY); \
        init_function_maps(); \
    } while(0)

#define FREE_FUNC_MAP \
    do { \
        free_hash_table(); \
    } while(0)


#define DEFINE_FUNC_MAP(method_uri, func_ptr) \
    do { \
        insert_item(method_uri, func_ptr); \
    } while(0)

#define FUNCTION_MAPS(...) \
    void init_function_maps() { \
        __VA_ARGS__ \
    }

#endif // __FUNCTIONS_MAP_H__