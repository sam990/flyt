#ifndef _GSCHED_H_
#define _GSCHED_H_

typedef struct _gsched_t {
    int (*init)(void);
    int (*retain)(int id);
    int (*release)(int id);
    int (*rm)(int id);
    void (*deinit)(void);
} gsched_t;

// gsched_t *sched;

// #define GSCHED_RETAIN sched->retain(rqstp->rq_xprt->xp_fd)
// #define GSCHED_RELEASE sched->release(rqstp->rq_xprt->xp_fd)

typedef struct _gsched_fixed_t {
    int (*init)(void);
    int (*shared)(void);
    int (*exclusive)(void);
    int (*release)(void);
    void (*deinit)(void);
} gsched_fixed_t;

gsched_fixed_t *sched;

#define GSCHED_RETAIN sched->shared()
#define GSCHED_RELEASE sched->release()
#define GSCHED_EXCLUSIVE sched->exclusive()



#endif //_GSCHED_H_
