#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<signal.h>

int tosuspendpid=0;
int interval=1500000;
int suspensiontime=1050000;
void childSignalHandler(int signum)
{
    if(signum==SIGUSR1)
    {
        kill(tosuspendpid,SIGSTOP);
        usleep(suspensiontime);
        kill(tosuspendpid,SIGCONT);
    }
}
int main()
{
    tosuspendpid=fork();
    printf("%d is the pid of process\n ",tosuspendpid);
    if(tosuspendpid<0)
    {
        perror("fork has failed\n");
    }
    if(tosuspendpid==0)
        {
             char *command[] = {"python3", "resnet50_pytorch.py", "--batch-size", "85", NULL};
            //  sleep(1);
             execvp("python3", command);
             perror("execvp has failed");
             exit(1);
        }
    int pid=fork();
    interval=1500000;
    suspensiontime=1050000;
    if(pid<0)
    {
        printf("fork has failed\n");
    }
    if(pid>0)
    {
        while(1)
        {
            usleep(interval);
            if(kill(pid,SIGUSR1)==-1)
            {
                perror("kill has failed");
                exit(1);
            }
        }
    }
    else
    {
        signal(SIGUSR1,childSignalHandler);
        int x=0;
        while(1){
            x=1;
        }
    }
    // wait();
    // wait();
    return 0;
}