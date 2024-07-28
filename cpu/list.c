#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "log.h"
#include "list.h"
#include "api-recorder.h"

#define INITIAL_CAPACITY 4

int list_init(list *l, size_t element_size)
{
    // Define and set mutex attribute directly
    pthread_mutexattr_t attr;

    // Initialize the attribute variable
    pthread_mutexattr_init(&attr);

    // Set the attribute to make the mutex recursive
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);

    //printf("This is line number %d\n", __LINE__);
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (element_size == 0LL) {
        LOGE(LOG_ERROR, "element_size of 0 does not make sense");
        return 1;
    }
    memset(l, 0, sizeof(list));
    if ((l->elements = malloc(INITIAL_CAPACITY*element_size)) == NULL) {
        LOGE(LOG_ERROR, "allocation failed");
        return 1;
    }
    pthread_mutex_init(&l->mutex, &attr);
    l->element_size = element_size;
    l->capacity = INITIAL_CAPACITY;
    l->length = 0LL;
    //printf("This is line number %d\n", __LINE__);

    return 0;
}

int list_init_capacity(list *l, size_t element_size, size_t capacity)
{
    // Define and set mutex attribute directly
    pthread_mutexattr_t attr;

    // Initialize the attribute variable
    pthread_mutexattr_init(&attr);

    // Set the attribute to make the mutex recursive
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);

    //printf("This is line number %d\n", __LINE__);
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (element_size == 0LL) {
        LOGE(LOG_ERROR, "element_size of 0 does not make sense");
        return 1;
    }
    size_t capacity_in_power_of_two;
    for (capacity_in_power_of_two = 1; capacity_in_power_of_two < capacity; capacity_in_power_of_two <<= 1);

    size_t new_capacity = capacity_in_power_of_two > INITIAL_CAPACITY ? capacity_in_power_of_two : INITIAL_CAPACITY;

    memset(l, 0, sizeof(list));
    if ((l->elements = malloc(new_capacity*element_size)) == NULL) {
        LOGE(LOG_ERROR, "allocation failed");
	l->capacity =  0LL;
	l->length =  0LL;
    	pthread_mutex_init(&l->mutex, &attr);
        return 1;
    }
    l->element_size = element_size;
    l->capacity = new_capacity;
    l->length = 0LL;
    pthread_mutex_init(&l->mutex, &attr);
    //printf("This is line number %d\n", __LINE__);

    return 0;
}

int list_resize(list *l, size_t new_capacity)
{
    //printf("This is line number %d\n", __LINE__);
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (new_capacity <= l->capacity) {
        return 0;
    }

    size_t rounded_capacity;
    for (rounded_capacity = 1; rounded_capacity < new_capacity; rounded_capacity <<= 1);
    void *nll;
    pthread_mutex_lock(&l->mutex);
    if ((nll = realloc(l->elements, rounded_capacity*l->element_size)) == NULL) {
        LOGE(LOG_ERROR, "allocation failed");
        pthread_mutex_unlock(&l->mutex);
        return 1;
    }
    l->capacity = new_capacity;
    l->elements = nll;
    pthread_mutex_unlock(&l->mutex);
    //printf("This is line number %d\n", __LINE__);
    return 0;
}

int list_free(list *l)
{
    //printf("This is line number %d\n", __LINE__);
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    pthread_mutex_lock(&l->mutex);
    l->capacity = 0;
    l->length = 0;
    free(l->elements);
    pthread_mutex_destroy(&l->mutex);
    //printf("This is line number %d\n", __LINE__);

    return 0;
}

/*
int list_free_elements(list *l)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    for (size_t i=0; i < l->length; ++i) {
        free(*(void**)list_get(l, i));
    }
    return 0;
}
*/

int list_append(list *l, void **new_element)
{
    //printf("This is line number %d\n", __LINE__);
    int ret = 0;
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    pthread_mutex_lock(&l->mutex);
    if (l->capacity == l->length) {
        void *nlist = realloc(l->elements, l->capacity*2*l->element_size);
        if (nlist== NULL) {
            LOGE(LOG_ERROR, "realloc failed.");
            /* the old pointer remains valid */
    	    pthread_mutex_unlock(&l->mutex);
            return 1;
        }
        l->elements = nlist;
        l->capacity *= 2;
    }
    if (new_element != NULL) {
        *new_element = list_get(l, l->length++);
    }
    pthread_mutex_unlock(&l->mutex);
    //printf("This is line number %d\n", __LINE__);

    return ret;
}

int list_append_copy(list *l, void *new_element)
{
    int ret = 0;
    void *elem;
    //printf("This is line number %d\n", __LINE__);
    if(new_element == NULL) {
        LOGE(LOG_ERROR, "new element is NULL");
        return 1;
    }

    pthread_mutex_lock(&l->mutex);
    if ( (ret = list_append(l, &elem)) != 0) {
        goto out;
    }
    memcpy(elem, new_element, l->element_size);
 out:
    pthread_mutex_unlock(&l->mutex);
    //printf("This is line number %d\n", __LINE__);
    return ret;
}

int list_at(list *l, size_t at, void **element)
{
    //printf("This is line number %d\n", __LINE__);
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    pthread_mutex_lock(&l->mutex);
    if (at >= l->length) {
        LOGE(LOG_ERROR, "accessing list out of bounds");
        pthread_mutex_unlock(&l->mutex);
        return 1;
    }
    if (element != NULL) {
        *element = list_get(l, at);
    }
    pthread_mutex_unlock(&l->mutex);
    //printf("This is line number %d\n", __LINE__);
    return 0;
}

inline void* list_get(list *l, size_t at) {
    return (l->elements+at*l->element_size);
}

int list_insert(list *l, size_t at, void *new_element)
{
    int val;
    //printf("This is line number %d\n", __LINE__);
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if(new_element == NULL) {
        LOGE(LOG_ERROR, "new element is NULL");
        return 1;
    }
    pthread_mutex_lock(&l->mutex);
    if (at > l->length) {
        LOGE(LOG_ERROR, "accessing list out of bounds");
        val = 1;
	goto out;
    }
    if (at == l->length) {

        val = list_append_copy(l, new_element);
	if (val != 0)
	    goto out;
    }

    if (list_append(l, NULL) != 0) {
        LOGE(LOG_ERROR, "error while lengthening list");
        val = 1;
	goto out;
    }
    memmove(list_get(l, at+1), list_get(l, at), (l->length-at -1)*l->element_size);

    if (new_element != NULL) {
        memcpy(list_get(l, at), new_element, l->element_size);
    }

    l->length += 1; //appending a NULL element does not increase list length
    val = 0;
out:
    //printf("This is line number %d\n", __LINE__);
    pthread_mutex_unlock(&l->mutex);
    return val;
}

int list_rm(list *l, size_t at)
{
    //printf("This is line number %d\n", __LINE__);
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    pthread_mutex_lock(&l->mutex);
    if (at >= l->length) {
        LOGE(LOG_ERROR, "accessing list out of bounds");
        pthread_mutex_unlock(&l->mutex);
        return 1;
    }
    if (at < l->length-1) {
        memmove(list_get(l, at), list_get(l, at+1), (l->length-1-at)*l->element_size);
    }
    l->length -= 1;
    //printf("This is line number %d\n", __LINE__);
    pthread_mutex_unlock(&l->mutex);
    return 0;
}
