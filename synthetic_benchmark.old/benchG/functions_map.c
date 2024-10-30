#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "functions_map.h"

typedef struct {
    char* method_uri;
    api_function method_function;
} hash_table_node;


typedef struct {
    int capacity;
    int curr_size;
    hash_table_node *table;
} hash_table;

static hash_table* method_uri_table;

static int check_prime(unsigned long num) {
    if (num <= 2 || (num % 2) == 0) {
        return 0;
    }

    for (int i = 3; i*i <= num; i+= 2) {
        if (num % i == 0) {
            return 0;
        }
    }
    return 1;
}

static unsigned long getUpperPrimeNumber(unsigned long num) {
    while (!check_prime(num)) {
        num++;
    }
    return num;
}

// hash function
// djb2 algorithm
static unsigned long hash(const char* str) {
    unsigned long hash = 5381;

    while(*str) {
        hash = ((hash << 5) + hash) + ((int)*str);
        str++;
    }
    return hash;
}

static unsigned long key(unsigned long hash) {
    return hash % method_uri_table->capacity;
}


// inserting a new entry 
int insert_item(const char* method_uri, api_function method_func_ptr) {
    if (method_uri_table->capacity <= method_uri_table->curr_size) {
        // return error as table is full and does not supports resizing as
        // of now
        return 0;
    }

    // get the hashkey
    unsigned long hash_key = key(hash(method_uri));

    int index = hash_key;

    while((method_uri_table->table[index].method_uri) != NULL) {
        index = (index + 1) % method_uri_table->capacity;
    }

    method_uri_table->table[index].method_uri = strdup(method_uri);
    method_uri_table->table[index].method_function = method_func_ptr;
    
    if (method_uri_table->table[index].method_uri == NULL) {
        // failed to allocate string
        return 0;
    }

    method_uri_table->curr_size++;
    return 1;
}

void init_hash_table(int min_capacity) {
    method_uri_table = malloc(sizeof(hash_table));
    method_uri_table->capacity = getUpperPrimeNumber(min_capacity);
    method_uri_table->curr_size = 0;

    method_uri_table->table = calloc(method_uri_table->capacity, sizeof(hash_table_node));
}


void free_hash_table() {
    free(method_uri_table->table);
    free(method_uri_table);
    method_uri_table = NULL;
}

int is_table_initialized(void) {
    return method_uri_table != NULL;
}


// Returns NULL if not found in hash table
api_function get_method_function(const char* method_str) {
    unsigned long hash_key = key(hash(method_str));

    int index = hash_key;

    while (method_uri_table->table[index].method_uri != NULL) {
        if (!strcmp(method_str, method_uri_table->table[index].method_uri)) {
            return method_uri_table->table[index].method_function;
        }
        index++;
    }
    // not found
    return NULL;
}
