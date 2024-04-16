#include "gsched.h"
#include <pthread.h>



pthread_rwlock_t rwlock;



int gsched_fixed_init(void)
{
    pthread_rwlock_init(&rwlock, NULL);
    return 0;
}

int gsched_fixed_shared(void)
{
    pthread_rwlock_rdlock(&rwlock);
    return 0;
}

int gsched_fixed_exclusive(void)
{
    pthread_rwlock_wrlock(&rwlock);
    return 0;
}


int gsched_fixed_release(void)
{
    pthread_rwlock_unlock(&rwlock);
    return 0;
}

void gsched_fixed_deinit(void)
{
    pthread_rwlock_destroy(&rwlock);
}

gsched_fixed_t sched_fixed = {
    .init = gsched_fixed_init,
    .shared = gsched_fixed_shared,
    .exclusive = gsched_fixed_exclusive,
    .release = gsched_fixed_release,
    .deinit = gsched_fixed_deinit,
};


