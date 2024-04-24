#ifndef _API_RECODER_H_
#define _API_RECODER_H_

#include <stdint.h>
#include "list.h"

#ifdef WITH_RECORDER
#define RECORD_VOID_API \
    api_record_t *record; \
    if (list_append(&api_records, (void**)&record) != 0) { \
        LOGE(LOG_ERROR, "list allocation failed."); \
    } \
    record->function = rqstp->rq_proc; \
    record->arg_size = 0; \
    record->arguments = NULL; \
    record->data_size = 0; \
    record->data = NULL;
#define RECORD_API(ARG_TYPE) \
    api_record_t *record; \
    ARG_TYPE *arguments; \
    if (list_append(&api_records, (void**)&record) != 0) { \
        LOGE(LOG_ERROR, "list allocation failed."); \
    } \
    if ( (arguments = malloc(sizeof(ARG_TYPE))) == NULL) { \
        LOGE(LOG_ERROR, "list arguments allocation failed"); \
    } \
    record->function = rqstp->rq_proc; \
    record->arg_size = sizeof(ARG_TYPE); \
    record->arguments = arguments; \
    record->data_size = 0; \
    record->data = NULL;
#define RECORD_API_STR(ARG_TYPE, NSTRS) \
    api_record_t *record; \
    ARG_TYPE *arguments; \
    if (list_append(&api_records, (void**)&record) != 0) { \
        LOGE(LOG_ERROR, "list allocation failed."); \
    } \
    if ( (arguments = malloc(sizeof(ARG_TYPE))) == NULL) { \
        LOGE(LOG_ERROR, "list arguments allocation failed"); \
    } \
    if ( (record->str_args = malloc(sizeof(char*) * NSTRS)) == NULL) { \
        LOGE(LOG_ERROR, "list str_args allocation failed"); \
    } \
    record->function = rqstp->rq_proc; \
    record->arg_size = sizeof(ARG_TYPE); \
    record->arguments = arguments; \
    record->data_size = 0; \
    record->data = NULL; \
    record->str_args_num = NSTRS; \
    record->str_arg_counter = 0;
#define RECORD_RESULT(TYPE, RES) \
    record->result.TYPE = RES
#define RECORD_SINGLE_ARG(ARG) \
    *arguments = ARG
#define RECORD_ARG(NUM, ARG) \
    arguments->arg##NUM = ARG
#define RECORD_ARG_STR(NUM, ARG) \
    record->str_args[record->str_arg_counter] = strdup(ARG); \
    arguments->arg##NUM = record->str_args[record->str_arg_counter++];
#define RECORD_NARG(ARG) \
    arguments->ARG = ARG
#define RECORD_DATA(SIZE, PTR) \
    record->data_size = SIZE; \
    record->data = malloc(SIZE); \
    memcpy(record->data, PTR, SIZE);
#else
#define RECORD_API(ARG_TYPE) 
#define RECORD_RESULT(TYPE, RES)
#define RECORD_ARG(NUM, ARG)
#define RECORD_SINGLE_ARG(ARG)
#endif //WITH_RECORDER

#include <stdio.h>
#include "cpu_rpc_prot.h"

typedef struct api_record {
    unsigned int function;
    size_t arg_size;
    void *arguments;
    union {
        uint64_t u64;
        void* ptr;
        int integer;
        ptr_result ptr_result_u;
        sz_result sz_result_u;
    } result;
    void *data;
    size_t data_size;
    void **str_args;
    size_t str_args_num;
    size_t str_arg_counter;
} api_record_t;
extern list api_records;


void api_records_free(void);
void api_records_print(void);
void api_records_print_records(api_record_t *record);

#endif
