/* Copyright (c) 2024-2026 SynerG Lab, IITB */

// custom_clnt_call.c
#include <pthread.h>
#include <rpc/rpc.h>
#include <sys/time.h>

pthread_mutex_t lock;
static int initialized = 0;

// Your custom clnt_call implementation that locks the mutex
enum clnt_stat clnt_call(CLIENT *clnt, unsigned long procnum,
                         xdrproc_t inproc, char *in,
                         xdrproc_t outproc, char *out,
                         struct timeval tout) {
    if(initialized == 0) {
	    // Define and set mutex attribute directly
	    pthread_mutexattr_t attr;

	    // Initialize the attribute variable
	    pthread_mutexattr_init(&attr);

	    // Set the attribute to make the mutex recursive
	    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);

	    pthread_mutex_init(&lock, &attr);
    }
    initialized = 1;
    // Lock the mutex
    pthread_mutex_lock(&lock);
    
    // Call the original clnt_call function (use dlsym to get the symbol address)
    static enum clnt_stat (*original_clnt_call)(CLIENT *, unsigned long, 
                                                xdrproc_t, char *, 
                                                xdrproc_t, char *, 
                                                struct timeval);
    if (!original_clnt_call) {
        original_clnt_call = (enum clnt_stat (*)(CLIENT *, unsigned long, 
                                                 xdrproc_t, char *, 
                                                 xdrproc_t, char *, 
                                                 struct timeval)) 
                            dlsym(RTLD_NEXT, "clnt_call");
    }
    
    // Call the original clnt_call
    enum clnt_stat status = original_clnt_call(clnt, procnum, inproc, in,
                                              outproc, out, tout);
    
    // Unlock the mutex
    pthread_mutex_unlock(&lock);
    
    return status;
}

